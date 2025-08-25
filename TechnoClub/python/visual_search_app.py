# visual_search_app.py
import os
import sqlite3
import json
import base64
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'visual_search.db'   # local sqlite for storing sample images/features
SAMPLE_IMAGES_FOLDER = 'sample_images'

# MySQL configuration - update with your DB credentials
MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'krrish@001#400',
    'database': 'fashioondb'
}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_IMAGES_FOLDER, exist_ok=True)


class VisualSearchDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
        self.populate_sample_data()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                title TEXT,
                description TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                color_histogram TEXT,
                feature_vector TEXT,
                dominant_colors TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query_image_path TEXT,
                results_count INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def populate_sample_data(self):
        """Add sample images to database if empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM images")
        count = cursor.fetchone()[0]

        if count == 0:
            sample_data = [
                {
                    'filename': 'mountain_landscape.png',
                    'title': 'Mountain Landscape',
                    'description': 'Beautiful mountain scenery with snow peaks',
                    'colors': [(70, 130, 200), (100, 150, 100), (200, 200, 200)]
                },
                {
                    'filename': 'ocean_waves.png',
                    'title': 'Ocean Waves',
                    'description': 'Peaceful ocean waves on a sunny day',
                    'colors': [(30, 144, 255), (0, 191, 255), (176, 196, 222)]
                },
                {
                    'filename': 'forest_path.png',
                    'title': 'Forest Path',
                    'description': 'Winding path through a green forest',
                    'colors': [(34, 139, 34), (107, 142, 35), (85, 107, 47)]
                },
                {
                    'filename': 'city_skyline.png',
                    'title': 'City Skyline',
                    'description': 'Modern city skyline at sunset',
                    'colors': [(105, 105, 105), (169, 169, 169), (255, 140, 0)]
                },
                {
                    'filename': 'desert_sunset.png',
                    'title': 'Desert Sunset',
                    'description': 'Golden sunset over sand dunes',
                    'colors': [(255, 165, 0), (255, 69, 0), (210, 180, 140)]
                }
            ]

            for data in sample_data:
                img_path = self.create_sample_image(data)
                features = self.extract_features(img_path)

                cursor.execute('''
                    INSERT INTO images (filename, file_path, title, description, 
                                     file_size, width, height, color_histogram, 
                                     feature_vector, dominant_colors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['filename'],
                    img_path,
                    data['title'],
                    data['description'],
                    features['file_size'],
                    features['width'],
                    features['height'],
                    json.dumps(features['color_histogram']),
                    json.dumps(features['feature_vector']),
                    json.dumps(features['dominant_colors'])
                ))

            conn.commit()
            print(f"Added {len(sample_data)} sample images to database")

        conn.close()

    def create_sample_image(self, data):
        """Create a sample image with dominant colors"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        colors = data['colors']
        height, width = img.shape[:2]

        for i, color in enumerate(colors):
            start_y = i * height // len(colors)
            end_y = (i + 1) * height // len(colors)
            img[start_y:end_y, :] = color[::-1]  # BGR format

        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img_path = os.path.join(SAMPLE_IMAGES_FOLDER, data['filename'])
        cv2.imwrite(img_path, img)

        return img_path

    def extract_features(self, image_path):
        """Extract visual features from image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = img.shape[:2]
        file_size = os.path.getsize(image_path)

        # Color histogram (simplified)
        hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])

        color_histogram = np.concatenate([
            hist_b.flatten(),
            hist_g.flatten(),
            hist_r.flatten()
        ]).tolist()

        # Simple feature vector (mean colors + texture approximation)
        mean_colors = np.mean(img.reshape(-1, 3), axis=0)

        # Basic texture feature (standard deviation)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture_feature = [np.std(gray)]

        feature_vector = np.concatenate([mean_colors, texture_feature]).tolist()

        # Dominant colors (simple mean of random sample)
        img_sample = img.reshape(-1, 3)
        sample_size = min(1000, len(img_sample))
        sample_indices = np.random.choice(len(img_sample), sample_size, replace=False)
        sample_pixels = img_sample[sample_indices]

        dominant_colors = []
        for i in range(3):
            dominant_colors.append(np.mean(sample_pixels, axis=0).astype(int).tolist())

        return {
            'width': width,
            'height': height,
            'file_size': file_size,
            'color_histogram': color_histogram,
            'feature_vector': feature_vector,
            'dominant_colors': dominant_colors
        }

    def search_similar_images(self, query_features, limit=10):
        """Search for similar images based on features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, filename, file_path, title, description, 
                   color_histogram, feature_vector
            FROM images
        ''')

        results = []
        query_vector = np.array(query_features['feature_vector']).reshape(1, -1)
        query_hist = np.array(query_features['color_histogram']).reshape(1, -1)

        for row in cursor.fetchall():
            try:
                db_vector = np.array(json.loads(row[6])).reshape(1, -1)
                db_hist = np.array(json.loads(row[5])).reshape(1, -1)

                # If lengths mismatch, fallback to simple distance checks
                if db_vector.shape[1] != query_vector.shape[1]:
                    continue

                feature_similarity = cosine_similarity(query_vector, db_vector)[0][0]
                hist_similarity = cosine_similarity(query_hist, db_hist)[0][0]

                combined_similarity = (feature_similarity * 0.6 + hist_similarity * 0.4)
                similarity_percentage = max(0, combined_similarity * 100)

                img_b64 = self.image_to_base64(row[2])

                results.append({
                    'id': row[0],
                    'filename': row[1],
                    'title': row[3],
                    'description': row[4],
                    'similarity': round(similarity_percentage, 1),
                    'image_data': img_b64
                })
            except Exception as e:
                print(f"Error processing image {row[1]}: {e}")
                continue

        results.sort(key=lambda x: x['similarity'], reverse=True)
        conn.close()

        return results[:limit]

    def image_to_base64(self, image_path):
        """Convert image to base64 string for web display"""
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                b64_string = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/png;base64,{b64_string}"
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None


# Initialize database
db = VisualSearchDB(DATABASE_PATH)


def fetch_items_from_mysql(limit=20):
    """
    Fetch items from MySQL 'items' table (provided by you).
    Returns a list of dicts: id, retailer_id, item_name, price, description.
    """
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, retailer_id, item_name, price, description FROM items LIMIT %s", (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return []


@app.route('/')
def index():
    """Serve the main page (without static files)"""
    # Optionally fetch some items from MySQL to display on the landing page
    items = fetch_items_from_mysql(limit=5)
    item_html = ""
    if items:
        item_html = "<h3>Items from MySQL</h3><ul>"
        for it in items:
            item_html += f"<li>{it['item_name']} — ₹{it['price']} — {it['description'][:60]}...</li>"
        item_html += "</ul>"

    # Serve the large HTML page (same as you had). For brevity reuse your earlier template.
    # NOTE: For maintainability move this to templates/ and static/ in production.
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Visual Search - Local DB</title>
      <style>/* You can paste your CSS here; omitted for brevity */ body{font-family:Arial,Helvetica,sans-serif;padding:20px;} .container{max-width:1100px;margin:0 auto;background:#fff;padding:20px;border-radius:12px;} </style>
    </head>
    <body>
      <div class="container">
        <h1>🔍 Visual Search</h1>
        <p>Upload an image to search local sample images. Flask talks to MySQL for item metadata.</p>
        {{item_html|safe}}
        <form id="uploadForm" enctype="multipart/form-data">
          <input type="file" name="image" id="imageInput" accept="image/*" />
          <button type="button" id="searchBtn">Search Similar Images</button>
        </form>
        <div id="results"></div>
      </div>

      <script>
      const btn = document.getElementById('searchBtn');
      btn.addEventListener('click', async () => {
        const input = document.getElementById('imageInput');
        if (!input.files || !input.files[0]) return alert('Select an image');
        const fd = new FormData();
        fd.append('image', input.files[0]);
        document.getElementById('results').innerHTML = 'Searching...';
        try {
          const res = await fetch('/search', { method: 'POST', body: fd });
          const data = await res.json();
          if (data.success) {
            let html = '<h3>Results</h3><div>';
            data.results.forEach(r => {
              html += `<div style="display:inline-block;margin:10px;border:1px solid #ddd;padding:8px;border-radius:6px;"><img src="${r.image_data}" style="width:140px;height:100px;object-fit:cover;display:block;" /><div>${r.title}</div><div>${r.similarity}%</div></div>`;
            });
            html += '</div>';
            document.getElementById('results').innerHTML = html;
          } else {
            document.getElementById('results').innerText = 'Error: ' + (data.error || 'unknown');
          }
        } catch (err) {
          document.getElementById('results').innerText = 'Connection error. Is Flask running?';
        }
      });
      </script>
    </body>
    </html>
    """, item_html=item_html)


@app.route('/search', methods=['POST'])
def search_images():
    """Handle image search requests"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)

        query_features = db.extract_features(temp_path)
        similar_images = db.search_similar_images(query_features, limit=8)

        # Optionally: you can also search MySQL items and return them alongside visual results
        mysql_items = fetch_items_from_mysql(limit=5)

        try:
            os.remove(temp_path)
        except OSError:
            pass

        return jsonify({
            'success': True,
            'results': similar_images,
            'items': mysql_items,
            'query_features': {
                'width': query_features['width'],
                'height': query_features['height'],
                'dominant_colors': query_features['dominant_colors']
            }
        })

    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/stats')
def get_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM search_history")
        search_count = cursor.fetchone()[0]
        conn.close()
        return jsonify({'image_count': image_count, 'search_count': search_count})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("🚀 Starting Visual Search Application on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
