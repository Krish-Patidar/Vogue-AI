<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Search - Local Database</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5rem;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            transform: translateY(-2px);
        }

        .upload-icon {
            font-size: 4rem;
            color: #667eea;
            margin-bottom: 20px;
        }

        .upload-text {
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }

        #fileInput {
            display: none;
        }

        .preview-container {
            margin: 20px 0;
            text-align: center;
        }

        .preview-image {
            max-width: 200px;
            max-height: 200px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        .search-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            margin: 20px auto;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .search-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }

        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .results-container {
            margin-top: 30px;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .result-item {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .result-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }

        .result-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }

        .result-info {
            padding: 15px;
        }

        .result-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }

        .result-description {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 10px;
        }

        .result-similarity {
            color: #667eea;
            font-weight: bold;
            font-size: 1.1rem;
        }

        .loading {
            text-align: center;
            color: #667eea;
            font-size: 1.1rem;
            margin: 20px 0;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #ffe6e6;
            color: #d63031;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }

        .success {
            background: #e6ffe6;
            color: #00b894;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }

        .db-info {
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            text-align: center;
        }

        .db-info h3 {
            color: #333;
            margin-bottom: 10px;
        }

        .db-info p {
            color: #666;
            margin-bottom: 10px;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
        }

        .stat-item {
            text-align: center;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }

        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Visual Search - Local Database</h1>
        
        <div class="db-info">
            <h3>📊 Database Status</h3>
            <p>Using local SQLite database with sample images</p>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number" id="imageCount">5</div>
                    <div class="stat-label">Images in DB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="searchCount">0</div>
                    <div class="stat-label">Searches Performed</div>
                </div>
            </div>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Drop an image here or click to browse</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <div class="preview-container" id="previewContainer" style="display: none;">
            <img id="previewImage" class="preview-image" alt="Preview">
        </div>

        <button class="search-btn" id="searchBtn" disabled>Search Similar Images</button>

        <div class="results-container" id="resultsContainer"></div>
    </div>

   
</body>
</html>