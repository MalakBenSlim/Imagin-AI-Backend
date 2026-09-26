import os, io, base64, json
from datetime import datetime
 
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import requests
 
from database import (
    init_pool, create_tables,
    insert_user, get_user_by_email, get_user_by_id,
    save_analysis, get_user_analyses
)
 
load_dotenv()
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img_pil):
    """Détecte et recadre le visage principal. Retourne l'image entière si aucun visage détecté."""
    arr = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return img_pil
    # Prendre le plus grand visage détecté
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Ajouter une marge de 20%
    margin = int(0.2 * max(w, h))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(arr.shape[1], x + w + margin)
    y1 = min(arr.shape[0], y + h + margin)
    return Image.fromarray(arr[y0:y1, x0:x1])
# =============================================================================
# CHARGEMENT DES MODÈLES IA (optionnel — désactivé si fichiers absents)
# =============================================================================
 
ARTIFY_MODEL_PATH  = os.getenv('ARTIFY_MODEL_PATH',  'saved_models/G_AB.pth')
INSIGHT_MODEL_PATH = os.getenv('INSIGHT_MODEL_PATH', 'saved_models/model_FINAL_CORRECT.pt')
 
# Micro-service STYLE ME (TensorFlow tourne dans un venv Python 3.11 séparé)
STYLEME_SERVICE_URL = None
DEVICE = 'cpu'
 
# ── PyTorch (Artify + Insight) ────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ PyTorch {torch.__version__} disponible — device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch non installé — Artify et Insight en mode mock")
 
# ── Classifieur vêtements (ResNet18 ImageNet pré-entraîné) ────────────────────
clothing_model = None
if TORCH_AVAILABLE:
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        _weights = ResNet18_Weights.IMAGENET1K_V1
        clothing_model = resnet18(weights=_weights).to(DEVICE)
        clothing_model.eval()
        _imagenet_classes = _weights.meta["categories"]
        print("✅ CLOTHING — ResNet18 ImageNet chargé")
    except Exception as e:
        print(f"❌ CLOTHING — erreur chargement : {e}")

# Mapping classes ImageNet (mots-clés) → catégories CLOTHING_CATEGORIES
IMAGENET_TO_CLOTHING = {
    "jersey": "T-shirt", "t-shirt": "T-shirt",
    "shirt": "Chemise", "Windsor tie": "Chemise",
    "gown": "Robe", "overskirt": "Robe",
    "trouser": "Pantalon", "jean": "Pantalon",
    "swimming trunks": "Short", "miniskirt": "Jupe",
    "suit": "Veste", "academic gown": "Veste",
    "trench coat": "Manteau", "fur coat": "Manteau",
    "cardigan": "Pull", "sweater": "Pull", "wool": "Pull",
    "kimono": "Combinaison", "abaya": "Combinaison",
}

def classify_clothing(img_pil):
    """Classifie le vêtement via ResNet18 ImageNet + mapping vers CLOTHING_CATEGORIES."""
    tfms = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tfms(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(clothing_model(tensor), dim=1)[0].cpu()
    top_idx = torch.topk(probs, 20).indices.tolist()
    for idx in top_idx:
        label = _imagenet_classes[idx]
        for keyword, category in IMAGENET_TO_CLOTHING.items():
            if keyword.lower() in label.lower():
                return category, float(probs[idx]) * 100
    return "T-shirt", float(probs[top_idx[0]]) * 100  # fallback
# ── Détection du micro-service STYLE ME ───────────────────────────────────────
STYLEME_SERVICE_AVAILABLE = False

print("🟡 STYLE ME — mode backend local (5000)")
 
# =============================================================================
# ARCHITECTURES DES MODÈLES
# =============================================================================
 
if TORCH_AVAILABLE:
 
    class ResidualBlock(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features), nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3),
                nn.InstanceNorm2d(in_features),
            )
        def forward(self, x): return x + self.block(x)
 
    class Generator(nn.Module):
        def __init__(self, input_shape=(3, 256, 256), num_residual_blocks=9):
            super().__init__()
            ch = input_shape[0]
            m = [nn.ReflectionPad2d(3), nn.Conv2d(ch, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True)]
            inf, outf = 64, 128
            for _ in range(2):
                m += [nn.Conv2d(inf, outf, 3, stride=2, padding=1), nn.InstanceNorm2d(outf), nn.ReLU(True)]
                inf, outf = outf, outf * 2
            for _ in range(num_residual_blocks): m += [ResidualBlock(inf)]
            outf = inf // 2
            for _ in range(2):
                m += [nn.ConvTranspose2d(inf, outf, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(outf), nn.ReLU(True)]
                inf, outf = outf, outf // 2
            m += [nn.ReflectionPad2d(3), nn.Conv2d(64, ch, 7), nn.Tanh()]
            self.model = nn.Sequential(*m)
        def forward(self, x): return self.model(x)
 
    class EmotionEfficientNet(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            try:
                import timm
                self.backbone = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0)
                num_features  = self.backbone.num_features
            except ImportError:
                from torchvision.models import efficientnet_b2
                base = efficientnet_b2(weights=None)
                self.backbone = nn.Sequential(*list(base.children())[:-1])
                num_features  = 1408
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features), nn.Dropout(0.3),
                nn.Linear(num_features, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )
        def forward(self, x):
            feats = self.backbone(x)
            if feats.dim() > 2: feats = feats.mean([-2, -1])
            return self.classifier(feats)
 
# =============================================================================
# CHARGEMENT DES MODÈLES
# =============================================================================
 
artify_model   = None
insight_model  = None
 
# ── Artify ────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE and os.path.exists(ARTIFY_MODEL_PATH):
    try:
        artify_model = Generator().to(DEVICE)
        artify_model.load_state_dict(torch.load(ARTIFY_MODEL_PATH, map_location=DEVICE))
        artify_model.eval()
        print(f"✅ ARTIFY  — modèle chargé ({DEVICE})")
    except Exception as e:
        print(f"❌ ARTIFY  — erreur chargement : {e}")
else:
    print(f"⚠️  ARTIFY  — mode mock (fichier absent ou PyTorch manquant)")
 
# ── Insight ───────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE and os.path.exists(INSIGHT_MODEL_PATH):
    try:
        checkpoint  = torch.load(INSIGHT_MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict  = checkpoint.get('model_state_dict', checkpoint)
        insight_model = EmotionEfficientNet(num_classes=7).to(DEVICE)
        insight_model.load_state_dict(state_dict, strict=False)
        insight_model.eval()
        print(f"✅ INSIGHT — modèle chargé ({DEVICE})")
    except Exception as e:
        print(f"❌ INSIGHT — erreur chargement : {e}")
else:
    print(f"⚠️  INSIGHT — mode mock (fichier absent ou PyTorch manquant)")
 
# =============================================================================
# DONNÉES MÉTIER
# =============================================================================
 
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_FR = {
    'Angry': 'Colère', 'Disgust': 'Dégoût', 'Fear': 'Peur',
    'Happy': 'Joie', 'Neutral': 'Neutre', 'Sad': 'Tristesse', 'Surprise': 'Surprise',
}
EMOTION_META = {
    'Angry'   : {'emoji': '😠', 'color': '#e05555', 'desc': "Une émotion intense — la frustration ou l'irritation dominent."},
    'Disgust' : {'emoji': '🤢', 'color': '#7ab648', 'desc': "Une réaction de répulsion ou de désapprobation forte."},
    'Fear'    : {'emoji': '😨', 'color': '#8a6cc9', 'desc': "Une émotion défensive face à une menace perçue."},
    'Happy'   : {'emoji': '😄', 'color': '#f5c542', 'desc': "La joie rayonne — une émotion positive et communicative."},
    'Neutral' : {'emoji': '😐', 'color': '#8899aa', 'desc': "Aucune émotion dominante — un état calme et équilibré."},
    'Sad'     : {'emoji': '😢', 'color': '#6c9ec9', 'desc': "La tristesse transparaît — une émotion de mélancolie."},
    'Surprise': {'emoji': '😲', 'color': '#f5a342', 'desc': "Une réaction inattendue — l'étonnement domine."},
}
 
CLOTHING_CATEGORIES = ["T-shirt","Chemise","Robe","Pantalon","Short","Veste","Manteau","Pull","Jupe","Combinaison"]
STYLE_MAP = {
    "T-shirt"    : {"style": "Casual",   "emoji": "👕", "tips": ["Jean slim", "Sneakers blanches", "Casquette"]},
    "Chemise"    : {"style": "Smart",    "emoji": "👔", "tips": ["Pantalon chino", "Derby cuir", "Montre sobre"]},
    "Robe"       : {"style": "Élégant",  "emoji": "👗", "tips": ["Escarpins", "Sac à main", "Bijoux discrets"]},
    "Pantalon"   : {"style": "Casual",   "emoji": "👖", "tips": ["T-shirt blanc", "Mocassins", "Ceinture cuir"]},
    "Short"      : {"style": "Sport",    "emoji": "🩳", "tips": ["T-shirt technique", "Baskets running", "Casquette sport"]},
    "Veste"      : {"style": "Smart",    "emoji": "🧥", "tips": ["Col roulé", "Chino beige", "Chelsea boots"]},
    "Manteau"    : {"style": "Chic",     "emoji": "🧣", "tips": ["Pull fin", "Bottines", "Écharpe cachemire"]},
    "Pull"       : {"style": "Casual",   "emoji": "🧶", "tips": ["Jean droit", "Boots", "Tote bag"]},
    "Jupe"       : {"style": "Élégant",  "emoji": "👗", "tips": ["Chemisier rentré", "Sandales", "Pochette"]},
    "Combinaison": {"style": "Tendance", "emoji": "✨", "tips": ["Sandales plates", "Sac paille", "Lunettes soleil"]},
}
 
# =============================================================================
# FLASK + CORS + JWT
# =============================================================================
 
app = Flask(__name__)
 
app.config['JWT_SECRET_KEY']       = os.getenv('JWT_SECRET_KEY', 'dev_secret_change_in_prod')
app.config['UPLOAD_FOLDER']        = 'uploads'
app.config['RESULT_FOLDER']        = 'results'
app.config['MAX_CONTENT_LENGTH']   = 16 * 1024 * 1024
 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
 
CORS(app,
     origins=["http://localhost:4200"],
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True)
 
jwt = JWTManager(app)
 
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        r = make_response()
        r.headers["Access-Control-Allow-Origin"]      = "http://localhost:4200"
        r.headers["Access-Control-Allow-Headers"]     = "Content-Type, Authorization"
        r.headers["Access-Control-Allow-Methods"]     = "GET, POST, PUT, DELETE, OPTIONS"
        r.headers["Access-Control-Allow-Credentials"] = "true"
        return r, 200
 
# =============================================================================
# UTILITAIRES
# =============================================================================
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}
 
def validate_image(file):
    """Vérifie que le fichier est bien une image (contenu réel, pas juste l'extension)."""
    try:
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)
        return True
    except Exception:
        file.stream.seek(0)
        return False
 
def pil_to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode()
 
def dominant_colors(img, n=5):
    px  = np.array(img.resize((100, 100)).convert("RGB")).reshape(-1, 3).astype(float)
    rng = np.random.default_rng(42)
    c   = px[rng.choice(len(px), n, replace=False)]
    for _ in range(8):
        d  = np.linalg.norm(px[:, None] - c[None], axis=2)
        lb = np.argmin(d, axis=1)
        c  = np.array([px[lb == k].mean(axis=0) if (lb == k).any() else c[k] for k in range(n)])
    d  = np.linalg.norm(px[:, None] - c[None], axis=2)
    lb = np.argmin(d, axis=1)
    out = []
    for k in range(n):
        r, g, b = c[k].astype(int)
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "proportion": round(float((lb == k).sum() / len(lb)), 3)})
    return sorted(out, key=lambda x: -x["proportion"])
 
# =============================================================================
# AUTH — /api/signup  /api/login  /api/me
# =============================================================================
 
@app.route('/api/signup', methods=['POST', 'OPTIONS'])
def signup():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Données manquantes"}), 400
 
    username = data.get('username', '').strip()
    email    = data.get('email',    '').strip()
    password = data.get('password', '')
 
    if not username or not email or not password:
        return jsonify({"success": False, "message": "Tous les champs sont obligatoires"}), 400
 
    if len(password) < 6:
        return jsonify({"success": False, "message": "Mot de passe trop court (min 6 caractères)"}), 400
 
    if get_user_by_email(email):
        return jsonify({"success": False, "message": "Cet email est déjà utilisé"}), 409
 
    try:
        user_id = insert_user(username, email, generate_password_hash(password))
        token   = create_access_token(identity=str(user_id))
        return jsonify({
            "success": True,
            "message": "Compte créé avec succès",
            "token"  : token,
            "user"   : {"id": user_id, "username": username, "email": email}
        }), 201
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
 
 
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Données manquantes"}), 400
 
    email    = data.get('email',    '')
    password = data.get('password', '')
 
    user = get_user_by_email(email)
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"success": False, "message": "Email ou mot de passe incorrect"}), 401
 
    token = create_access_token(identity=str(user['id']))
    return jsonify({
        "success": True,
        "token"  : token,
        "user"   : {"id": user['id'], "username": user['username'], "email": user['email']}
    }), 200
 
 
@app.route('/api/me', methods=['GET'])
@jwt_required()
def me():
    """Retourne le profil de l'utilisateur connecté."""
    user_id = int(get_jwt_identity())
    user    = get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "message": "Utilisateur introuvable"}), 404
    return jsonify({"success": True, "user": dict(user)}), 200
 
 
@app.route('/api/history', methods=['GET'])
@jwt_required()
def history():
    """Retourne l'historique des analyses de l'utilisateur connecté."""
    user_id       = int(get_jwt_identity())
    analysis_type = request.args.get('type')   # ?type=artify | insight | styleme
    rows          = get_user_analyses(user_id, analysis_type, limit=20)
    analyses = []
    for row in rows:
        r = dict(row)
        if r.get('result'):
            try: r['result'] = json.loads(r['result'])
            except: pass
        analyses.append(r)
    return jsonify({"success": True, "analyses": analyses}), 200
 
# =============================================================================
# ARTIFY — POST /transform
# =============================================================================
 
@app.route('/transform', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)   # fonctionne avec ou sans token
def transform_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
 
    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Format invalide (PNG, JPG, JPEG, WEBP)'}), 400
 
    if not validate_image(file):
        return jsonify({'error': 'Le fichier ne semble pas être une image valide'}), 400
 
    try:
        uname  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        upath  = os.path.join(app.config['UPLOAD_FOLDER'], uname)
        file.save(upath)
        img_pil = Image.open(upath).convert("RGB")
 
        # ── Vrai modèle ──────────────────────────────────────────────────────
        if artify_model is not None and TORCH_AVAILABLE:
            artify_transforms = T.Compose([
                T.Resize((256, 256)), T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            tens = artify_transforms(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = artify_model(tens)
            out     = ((out.squeeze(0).cpu() * 0.5) + 0.5).clamp(0, 1)
            result_pil = T.ToPILImage()(out)
 
        # ── Mode mock (modèle absent) ─────────────────────────────────────────
        else:
            # Simule une transformation : teinte sépia + contraste
            arr = np.array(img_pil.resize((256, 256)), dtype=np.float32) / 255.0
            sepia = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            transformed = np.clip(arr @ sepia.T, 0, 1)
            result_pil  = Image.fromarray((transformed * 255).astype(np.uint8))
 
        # ── Sauvegarde + réponse ──────────────────────────────────────────────
        rfname = f"art_{uname}"
        result_pil.save(os.path.join(app.config['RESULT_FOLDER'], rfname))
 
        result_b64 = f"data:image/jpeg;base64,{pil_to_b64(result_pil)}"
 
        # Sauvegarder dans l'historique si connecté
        user_id = get_jwt_identity()
        if user_id:
            save_analysis(int(user_id), 'artify', json.dumps({
                'filename': uname, 'mock': artify_model is None
            }))
 
        return jsonify({
            'success' : True,
            'result'  : result_b64,
            'imageUrl': f'http://127.0.0.1:5000/results/{rfname}',
            'mock'    : artify_model is None,
        }), 200
 
    except Exception as e:
        print(f"🔥 ARTIFY: {e}")
        return jsonify({'error': str(e)}), 500
 
# =============================================================================
# INSIGHT — POST /insight
# =============================================================================
 
@app.route('/insight', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)
def insight():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
 
    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Format invalide (PNG, JPG, JPEG, WEBP)'}), 400
 
    if not validate_image(file):
        return jsonify({'error': 'Le fichier ne semble pas être une image valide'}), 400
 
    try:
        uname   = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        upath   = os.path.join(app.config['UPLOAD_FOLDER'], f"insight_{uname}")
        file.save(upath)
        img_pil = Image.open(upath).convert("RGB")
 
        # ── Vrai modèle ──────────────────────────────────────────────────────
        if insight_model is not None and TORCH_AVAILABLE:
            face_img = crop_face(img_pil)
            print(f"Image originale: {img_pil.size}, Image croppée: {face_img.size}")
            face_img.save("debug_face.jpg")
            insight_transforms = T.Compose([
                T.Resize((260, 260)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = insight_transforms(face_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(insight_model(tensor), dim=1)[0].cpu()
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx]) * 100
            probs_list = probs.tolist()
 
        # ── Mode mock ─────────────────────────────────────────────────────────
        else:
            probs_list = list(np.random.dirichlet(np.ones(7) * 0.5))
            pred_idx   = int(np.argmax(probs_list))
            confidence = float(probs_list[pred_idx]) * 100
 
        emotion = EMOTIONS[pred_idx]
        meta    = EMOTION_META[emotion]
 
        top3_idx = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)[:min(3, len(probs_list))]
        top3 = [
            {"emotion": EMOTIONS[i], "emotion_fr": EMOTION_FR[EMOTIONS[i]],
             "confidence": round(probs_list[i] * 100, 1), "emoji": EMOTION_META[EMOTIONS[i]]['emoji']}
            for i in top3_idx
        ]
        all_emotions = [
            {"emotion": EMOTIONS[i], "emotion_fr": EMOTION_FR[EMOTIONS[i]],
             "probability": round(probs_list[i] * 100, 1), "emoji": EMOTION_META[EMOTIONS[i]]['emoji']}
            for i in range(7)
        ]
 
        response_data = {
            'success'    : True,
            'emotion'    : emotion,
            'emotion_fr' : EMOTION_FR[emotion],
            'confidence' : round(confidence, 1),
            'emoji'      : meta['emoji'],
            'color'      : meta['color'],
            'description': meta['desc'],
            'top3'       : top3,
            'all_emotions': all_emotions,
            'imageBase64': f"data:image/jpeg;base64,{pil_to_b64(img_pil)}",
            'mock'       : insight_model is None,
        }
 
        user_id = get_jwt_identity()
        if user_id:
            save_analysis(int(user_id), 'insight', json.dumps({
                'emotion': emotion, 'confidence': round(confidence, 1), 'mock': insight_model is None
            }))
 
        return jsonify(response_data), 200
 
    except Exception as e:
        print(f"🔥 INSIGHT: {e}")
        return jsonify({'error': str(e)}), 500
 
# =============================================================================
# STYLE ME — POST /styleme
# =============================================================================
 
@app.route('/styleme', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True)
def style_me():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
 
    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Format invalide (PNG, JPG, JPEG, WEBP)'}), 400
 
    if not validate_image(file):
        return jsonify({'error': 'Le fichier ne semble pas être une image valide'}), 400
 
    try:
        uname   = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        upath   = os.path.join(app.config['UPLOAD_FOLDER'], f"sm_{uname}")
        file.save(upath)
        img_pil = Image.open(upath).convert("RGB")
 
        
        if clothing_model is not None and TORCH_AVAILABLE:
            category, confidence_pct = classify_clothing(img_pil)
            confidence = confidence_pct / 100
            mock_flag  = False
        else:
            category   = "T-shirt"
            confidence = 0.0
            mock_flag  = True

        sinfo = STYLE_MAP.get(category, {"style": "Moderne", "emoji": "👗", "tips": ["Accessoire tendance"]})
        top3  = []
        colors = dominant_colors(img_pil)
 
        response_data = {
            'success'        : True,
            'category'       : category,
            'confidence'     : round(confidence * 100, 1),
            'style'          : sinfo['style'],
            'emoji'          : sinfo['emoji'],
            'tips'           : sinfo['tips'],
            'top3'           : top3,
            'dominant_colors': colors,
            'imageBase64'    : f"data:image/jpeg;base64,{pil_to_b64(img_pil)}",
            'mock'           : mock_flag,
        }
 
        user_id = get_jwt_identity()
        if user_id:
            save_analysis(int(user_id), 'styleme', json.dumps({
                'category': category, 'confidence': round(confidence * 100, 1), 'mock': mock_flag
            }))
 
        return jsonify(response_data), 200
 
    except Exception as e:
        print(f"🔥 STYLEME: {e}")
        return jsonify({'error': str(e)}), 500
 
# =============================================================================
# UTILITAIRES
# =============================================================================
 
@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)
 
 
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status'        : 'ok',
        'device'        : DEVICE,
        'artify_loaded' : artify_model  is not None,
        'insight_loaded': insight_model is not None,
        'styleme_loaded': check_styleme_service(),
        'torch_available': TORCH_AVAILABLE,
        'styleme_service_url': STYLEME_SERVICE_URL,
    }), 200
 
# =============================================================================
# DÉMARRAGE
# =============================================================================
 
if __name__ == '__main__':
    init_pool()
    create_tables()
 
    print("\n🚀 IMAGIN.AI Backend — http://localhost:5000")
    print(f"   ARTIFY   : {'✅ modèle réel' if artify_model  else '🟡 mode mock'}")
    print(f"   INSIGHT  : {'✅ modèle réel' if insight_model else '🟡 mode mock'}")
    print(f"   STYLE ME : {'✅ modèle réel (micro-service)' if STYLEME_SERVICE_AVAILABLE else '🟡 mode mock'}\n")
 
    app.run(
        debug=bool(os.getenv('FLASK_DEBUG', 'true') == 'true'),
        port=int(os.getenv('FLASK_PORT', 5000))
    )
    
    
    
