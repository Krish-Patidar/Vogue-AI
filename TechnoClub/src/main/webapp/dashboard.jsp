<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>

<%@ page import="java.sql.*, com.fashionauth.DBConnection" %>

<%
    String userName = null;

    if (session != null) {   // directly use JSP implicit session
        userName = (String) session.getAttribute("userName");
    }

    if (userName == null) {
        response.sendRedirect("login.jsp");
    }
%>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VOGUE AI - Luxury Fashion Search Engine</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Montserrat:wght@300;400;500;600;700&family=Bebas+Neue&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --primary-black: #0a0a0a;
            --luxury-gold: #d4af37;
            --rose-gold: #E8B4B8;
            --deep-purple: #6B46C1;
            --electric-blue: #00D9FF;
            --neon-pink: #FF10F0;
            --soft-cream: #FFF8E7;
            --dark-gray: #1a1a1a;
            --gradient-luxury: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-gold: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
            --gradient-neon: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 52%, #2BFF88 90%);
            --gradient-dark: linear-gradient(135deg, #0F0C29 0%, #302B63 50%, #24243e 100%);
            --gradient-sunset: linear-gradient(135deg, #FA8BFF 0%, #FFD89B 100%);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Montserrat', sans-serif;
            background: #000;
            min-height: 100vh;
            color: #fff;
            overflow-x: hidden;
        }
        
        /* Animated Luxury Background */
        .luxury-bg {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -2;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 25%, #2d1b69 50%, #0f0f0f 75%, #000 100%);
            background-size: 400% 400%;
            animation: luxuryGradient 20s ease infinite;
        }
        
        .particles {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
        }
        
        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background: var(--luxury-gold);
            border-radius: 50%;
            animation: float-up 15s infinite linear;
            opacity: 0;
        }
        
        @keyframes float-up {
            0% {
                opacity: 0;
                transform: translateY(100vh) scale(0);
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                opacity: 0;
                transform: translateY(-100vh) scale(1.5);
            }
        }
        
        @keyframes luxuryGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Navbar Styles */
        .navbar {
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(212, 175, 55, 0.3);
            padding: 20px 0;
            transition: all 0.3s ease;
        }
        
        .navbar-brand {
            font-family: 'Bebas Neue', cursive;
            font-size: 2.5rem;
            letter-spacing: 3px;
            background: linear-gradient(90deg, var(--luxury-gold), #fff, var(--rose-gold));
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 3s linear infinite;
        }
        
        @keyframes shine {
            to {
                background-position: 200% center;
            }
        }
        
        .nav-link {
            color: #fff !important;
            margin: 0 20px;
            font-weight: 500;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 0.85rem;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .nav-link::before {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 2px;
            background: var(--luxury-gold);
            transition: width 0.3s ease;
        }
        
        .nav-link:hover {
            color: var(--luxury-gold) !important;
            transform: translateY(-2px);
        }
        
        .nav-link:hover::before {
            width: 100%;
        }
        /* User Avatar Styles */
.user-profile {
    padding: 0 !important;
    margin: 0 10px !important;
    display: flex;
    align-items: center;
}

.user-avatar {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    border: 2px solid var(--luxury-gold);
    transition: all 0.3s ease;
    object-fit: cover;
}

.user-avatar:hover {
    transform: scale(1.1);
    box-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
}

/* Cart Icon Styling */
.nav-link i.fa-shopping-cart {
    margin-right: 5px;
    color: var(--luxury-gold);
}
        /* Hero Section */
        .hero-section {
            padding: 120px 0 80px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 5rem;
            font-weight: 900;
            letter-spacing: -2px;
            line-height: 1.1;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #fff 0%, var(--luxury-gold) 50%, #fff 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 4s linear infinite, slideInDown 1s ease;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 300;
            letter-spacing: 3px;
            text-transform: uppercase;
            animation: fadeIn 1s ease 0.5s both;
        }
        
        @keyframes slideInDown {
            from {
                transform: translateY(-100px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Search Bar Luxury Style */
        .search-container {
            max-width: 900px;
            margin: 60px auto;
            position: relative;
            animation: fadeInUp 1s ease 0.8s both;
        }
        
        @keyframes fadeInUp {
            from {
                transform: translateY(30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .search-box {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 100px;
            padding: 25px 30px;
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            transition: all 0.4s ease;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        
        .search-box:hover {
            transform: translateY(-5px);
            border-color: var(--luxury-gold);
            box-shadow: 0 20px 60px rgba(212, 175, 55, 0.3);
        }
        
        .search-input {
            background: transparent;
            border: none;
            outline: none;
            flex: 1;
            padding: 10px 20px;
            font-size: 1.1rem;
            color: #fff;
            font-weight: 300;
        }
        
        .search-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        
        .search-btn {
            background: var(--gradient-gold);
            border: none;
            color: var(--primary-black);
            padding: 15px 40px;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(212, 175, 55, 0.4);
        }
        
        .search-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(212, 175, 55, 0.6);
        }
        
        /* Feature Cards - Luxury Glass Morphism */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
            padding: 60px 0;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 30px;
            padding: 40px 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.4s ease;
            cursor: pointer;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, var(--luxury-gold) 0%, transparent 70%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-15px) scale(1.02);
            background: rgba(212, 175, 55, 0.1);
            border-color: var(--luxury-gold);
            box-shadow: 0 30px 60px rgba(212, 175, 55, 0.4);
        }
        
        .feature-card:hover::before {
            opacity: 0.1;
        }
        
        .feature-icon {
            font-size: 3.5rem;
            margin-bottom: 25px;
            background: var(--gradient-gold);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            animation: pulse 3s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .feature-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            margin-bottom: 15px;
            color: #fff;
            font-weight: 700;
        }
        
        .feature-desc {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Stats Section - Luxury Style */
        .stats-section {
            padding: 80px 0;
            background: rgba(212, 175, 55, 0.03);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(212, 175, 55, 0.2);
            border-bottom: 1px solid rgba(212, 175, 55, 0.2);
            margin: 60px 0;
        }
        
        .stat-item {
            text-align: center;
            padding: 30px;
            position: relative;
        }
        
        .stat-number {
            font-family: 'Bebas Neue', cursive;
            font-size: 4rem;
            letter-spacing: 2px;
            background: var(--gradient-gold);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: block;
            animation: countUp 2s ease;
        }
        
        @keyframes countUp {
            from { 
                transform: scale(0) rotateY(180deg);
                opacity: 0;
            }
            to { 
                transform: scale(1) rotateY(0);
                opacity: 1;
            }
        }
        
        .stat-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: rgba(255, 255, 255, 0.5);
            margin-top: 10px;
        }
        
        /* Product Cards - Luxury Fashion Style */
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 40px;
            padding: 60px 0;
        }
        
        .product-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            transition: all 0.4s ease;
            cursor: pointer;
        }
        
        .product-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 30px 60px rgba(212, 175, 55, 0.3);
            border-color: var(--luxury-gold);
        }
        
        .product-img-container {
            position: relative;
            height: 400px;
            overflow: hidden;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d1b69 100%);
        }
        
        .product-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.6s ease;
            opacity: 0.9;
        }
        
        .product-card:hover .product-img {
            transform: scale(1.15);
            opacity: 1;
        }
        
        .product-badge {
            position: absolute;
            top: 20px;
            right: 20px;
            background: var(--gradient-gold);
            color: var(--primary-black);
            padding: 8px 20px;
            border-radius: 30px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            animation: slideInRight 0.5s ease;
        }
        
        @keyframes slideInRight {
            from {
                transform: translateX(100px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .product-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 100%);
            padding: 30px;
            transform: translateY(20px);
            opacity: 0;
            transition: all 0.4s ease;
        }
        
        .product-card:hover .product-overlay {
            transform: translateY(0);
            opacity: 1;
        }
        
        .product-info {
            padding: 25px;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
        }
        
        .product-brand {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--luxury-gold);
            margin-bottom: 8px;
        }
        
        .product-name {
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 15px;
        }
        
        .product-price {
            font-size: 1.8rem;
            font-weight: 300;
            color: var(--luxury-gold);
            font-family: 'Bebas Neue', cursive;
            letter-spacing: 1px;
        }
        
        .product-actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn-luxury {
            flex: 1;
            padding: 12px;
            background: transparent;
            border: 1px solid var(--luxury-gold);
            color: var(--luxury-gold);
            border-radius: 50px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .btn-luxury:hover {
            background: var(--luxury-gold);
            color: var(--primary-black);
            transform: translateY(-2px);
        }
        
        /* AI Assistant - Luxury Style */
        .ai-assistant-btn {
            position: fixed;
            bottom: 40px;
            right: 40px;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: var(--gradient-gold);
            color: var(--primary-black);
            border: none;
            cursor: pointer;
            box-shadow: 0 10px 40px rgba(212, 175, 55, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            transition: all 0.3s ease;
            animation: float 3s ease-in-out infinite;
            z-index: 1000;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-15px) rotate(5deg); }
        }
        
        .ai-assistant-btn:hover {
            transform: scale(1.15);
            box-shadow: 0 15px 50px rgba(212, 175, 55, 0.7);
        }
        
        /* Section Title */
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 60px;
            position: relative;
            color: #fff;
        }
        
        .section-title::after {
            content: '';
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 2px;
            background: var(--luxury-gold);
        }
        
        .section-subtitle {
            text-align: center;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            color: var(--luxury-gold);
            margin-bottom: 15px;
        }
        
        /* Filter Pills */
        .filter-pills {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        
        .filter-pill {
            padding: 10px 25px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 50px;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .filter-pill:hover {
            background: var(--luxury-gold);
            color: var(--primary-black);
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(212, 175, 55, 0.4);
        }
        /* Enhanced Product Grid Layout */
.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 35px;
    padding: 60px 0;
    max-width: 1400px;
    margin: 0 auto;
}
/* Footer */
.footer {
    background: rgba(10, 10, 10, 0.95);
    backdrop-filter: blur(20px);
    border-top: 1px solid rgba(212, 175, 55, 0.3);
    padding: 80px 0 40px;
    margin-top: 120px;
}

.footer-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 40px;
    margin-bottom: 60px;
}

.footer-section h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: var(--luxury-gold);
    margin-bottom: 20px;
    font-weight: 700;
}

.footer-section p, .footer-section a {
    color: rgba(255, 255, 255, 0.7);
    text-decoration: none;
    line-height: 1.8;
    font-size: 0.9rem;
    font-weight: 300;
    transition: color 0.3s ease;
}

.footer-section a:hover {
    color: var(--luxury-gold);
}

.footer-section ul {
    list-style: none;
    padding: 0;
}

.footer-section ul li {
    margin-bottom: 10px;
}

.social-links {
    display: flex;
    gap: 20px;
    margin-top: 20px;
}

.social-links a {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 45px;
    height: 45px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.3);
    border-radius: 50%;
    color: rgba(255, 255, 255, 0.7);
    font-size: 1.2rem;
    transition: all 0.3s ease;
}

.social-links a:hover {
    background: var(--luxury-gold);
    color: var(--primary-black);
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(212, 175, 55, 0.4);
}

.footer-bottom {
    text-align: center;
    padding-top: 40px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.footer-bottom p {
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.85rem;
    margin: 0;
}
/* Responsive Grid */
@media (max-width: 1200px) {
    .product-grid {
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 30px;
    }
}

@media (max-width: 768px) {
    .product-grid {
        grid-template-columns: 1fr;
        gap: 25px;
        padding: 40px 0;
    }
}
        /* Responsive Design */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 3rem;
            }
            
            .navbar-brand {
                font-size: 1.8rem;
            }
            
            .feature-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .product-grid {
                grid-template-columns: 1fr;
            }
            
            .stat-number {
                font-size: 3rem;
            }
        }
        /* Navbar Styles - Minimalist */
.navbar {
  background: rgba(10, 10, 10, 0.8);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(212, 175, 55, 0.3);
  padding: 12px 0;   /* reduced from 20px */
  transition: all 0.3s ease;
}

.navbar-brand {
  font-family: 'Bebas Neue', cursive;
  font-size: 2.2rem;   /* slightly smaller (was 2.5rem) */
  letter-spacing: 3px;
  background: linear-gradient(90deg, var(--luxury-gold), #fff, var(--rose-gold));
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shine 3s linear infinite;
}
        
    </style>
</head>
<body>
    <!-- Luxury Background -->
    <div class="luxury-bg"></div>
    <div class="particles" id="particles"></div>
    
    <!-- Navigation -->
   <!-- Navigation -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top">
    <div class="container">
        <a class="navbar-brand" href="main.jsp">
            <i class="fas fa-crown"></i> VOGUE AI
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item">
                    <a class="nav-link" href="dashboard.jsp">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="discover.jsp">Discover</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://ai-wardrobe-advisor.vercel.app/">Compare</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="wishlist.jsp">Wishlist</a>
                </li>
               
                <li class="nav-item">
                    <a class="nav-link" href="cart.jsp">
                        <i class="fas fa-shopping-cart"></i> Cart
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link user-profile" href="account.jsp">
                        <img src="./image/download.jpg" alt="User" class="user-avatar">
                    </a>
                </li>
            </ul>
        </div>
    </div>
</nav>
    
    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container">
         <h4>Welcome, <%= userName %>!</h4>
            <p class="hero-subtitle">AI-Powered Luxury Fashion Discovery</p>
            <h1 class="hero-title">Find Your<br>Signature Style</h1>
            
            <!-- Search Bar -->
            <div class="search-container">
                <div class="search-box">
                    <i class="fas fa-search" style="color: var(--luxury-gold); margin: 0 15px; font-size: 1.3rem;"></i>
                    <input type="text" class="search-input" placeholder="Search designer brands, latest trends, or upload your style...">
                    <button class="search-btn">
            <a href="http://127.0.0.1:5000/" style="color: white; text-decoration: none;"><i class="fas fa-sparkles"></i> Wardrobe AI</a>
                    </button>
                </div>
                
                <!-- Filter Pills -->
                <div class="filter-pills">
                    <div class="filter-pill">
                        <i class="fas fa-camera"></i>            <a href="visualsearch.jsp" style="color: white; text-decoration: none;"> Visual Search </a>
                    </div>
                    <div class="filter-pill">
                        <i class="fas fa-fire"></i>  <a href="trending.jsp" style="color: white; text-decoration: none;"> Trending </a>
                    </div>
                    <div class="filter-pill">
                        <i class="fas fa-gem"></i>  <a href="trending.jsp" style="color: white; text-decoration: none;"> Luxury </a>
                    </div>
                    <div class="filter-pill">
                        <i class="fas fa-tag"></i> <a href="premium.jsp" style="color: white; text-decoration: none;"> Explore Plans </a>
                    </div>
                    <div class="filter-pill">
                        <i class="fas fa-star"></i>  <a href="trending.jsp" style="color: white; text-decoration: none;"> New Arrivals </a>
                    </div>
                </div>
            </div>
        </div>
    </section>

   <!-- Curated Collection -->
<section class="container my-5">
    <p class="section-subtitle text-center">Curated by AI</p>
    <h2 class="section-title text-center mb-4">Today's Selection</h2>

    <div class="product-grid">
        <!-- Row 1 -->
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">AI Choice</span>
                <img src="https://tse1.explicit.bing.net/th/id/OIP.CTHou8Xmp_ZHYGyBtlukrwAAAA?rs=1&pid=ImgDetMain&o=7&rm=3" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-sparkles" style="color: var(--luxury-gold);"></i> 98% Style Match
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Anvi Ethnics</p>
                <h3 class="product-name">Kanjivaram Silk Saree</h3>
                <p class="product-price">₹749</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Trending</span>
                <img src="https://cdn0.weddingwire.in/article/6486/original/1920/jpg/116846-saree-poses-kavya-satheesan.jpeg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-fire" style="color: var(--luxury-gold);"></i> Trending Now
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Sabyasachi</p>
                <h3 class="product-name">Designer Bridal Saree</h3>
                <p class="product-price">₹12,500</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">New</span>
                <img src="https://cdn2.stylecraze.com/wp-content/uploads/2018/05/Kerala-Saree-With-Gold-Brocade-Blouse.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-star" style="color: var(--luxury-gold);"></i> Just Arrived
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Kerala Handlooms</p>
                <h3 class="product-name">Gold Brocade Kerala Saree</h3>
                <p class="product-price">₹2,850</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <!-- Row 2 -->
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Luxury</span>
                <img src="https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_750,h_750/global/846683/98/mod01/fnd/IND/fmt/png/Men's-Padded-Jacket" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-gem" style="color: var(--luxury-gold);"></i> Exclusive Edition
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Puma</p>
                <h3 class="product-name">Men's Padded Jacket</h3>
                <p class="product-price">₹8,999</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Sale</span>
                <img src="https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_750,h_750/global/535575/39/fnd/IND/fmt/png/Classics-Men's-Padded-Jacket" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-tag" style="color: var(--luxury-gold);"></i> 30% Off
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Puma</p>
                <h3 class="product-name">Classics Men's Padded Jacket</h3>
                <p class="product-price">₹6,299</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Limited</span>
                <img src="https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_500,h_500/global/672932/97/mod01/fnd/IND/fmt/png/Slub-Men's-Jacket" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-crown" style="color: var(--luxury-gold);"></i> Limited Edition
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Puma</p>
                <h3 class="product-name">Slub Men's Jacket</h3>
                <p class="product-price">₹7,499</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <!-- Row 3 -->
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Popular</span>
                <img src="https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_500,h_500/global/398846/02/sv01/fnd/IND/fmt/png/Speedcat-OG-Sneakers" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-thumbs-up" style="color: var(--luxury-gold);"></i> Most Popular
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Puma</p>
                <h3 class="product-name">Speedcat OG Sneakers</h3>
                <p class="product-price">₹9,999</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_500,h_500/global/403643/01/sv01/fnd/IND/fmt/png/PUMA-x-Squid-Game-Easy-Rider-Comfort-Sneakers" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Puma x Squid Game</p>
                <h3 class="product-name">Easy Rider Comfort Sneakers</h3>
                <p class="product-price">₹12,999</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://images.puma.com/image/upload/f_auto,q_auto,b_rgb:fafafa,w_500,h_500/global/402203/03/sv01/fnd/IND/fmt/png/RS-X-Geek-Advanced-Cushioning-Sneakers" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Puma</p>
                <h3 class="product-name">RS-X Geek Sneakers</h3>
                <p class="product-price">₹11,999</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Popular</span>
                <img src="https://m.media-amazon.com/images/I/91RcF1+wg3L.SY695.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-thumbs-up" style="color: var(--luxury-gold);"></i> Most Popular
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Kalini</p>
                <h3 class="product-name">Floral Print Anarkali Kurta</h3>
                <p class="product-price">₹1,299</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://m.media-amazon.com/images/I/61n+BjYI4XL.SY695.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Libas</p>
                <h3 class="product-name">Embroidered Kurta Set</h3>
                <p class="product-price">₹1,899</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://m.media-amazon.com/images/I/71qiwo36giL.SY695.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Indya</p>
                <h3 class="product-name">Block Print Kurta</h3>
                <p class="product-price">₹1,649</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Popular</span>
                <img src="https://m.media-amazon.com/images/I/71Ivda9vfNL.SY741.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-thumbs-up" style="color: var(--luxury-gold);"></i> Most Popular
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Aurelia</p>
                <h3 class="product-name">Straight Fit Kurta</h3>
                <p class="product-price">₹999</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://m.media-amazon.com/images/I/61STe02fQzL.SY741.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">W</p>
                <h3 class="product-name">Ethnic Printed Kurta</h3>
                <p class="product-price">₹1,499</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://m.media-amazon.com/images/I/51erncjGLuL.SY741.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Jaipur Kurti</p>
                <h3 class="product-name">Cotton Straight Kurta</h3>
                <p class="product-price">₹899</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Popular</span>
                <img src="https://m.media-amazon.com/images/I/61bF+gMRslL.SX522.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-thumbs-up" style="color: var(--luxury-gold);"></i> Most Popular
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Roadster</p>
                <h3 class="product-name">Men's Casual Shirt</h3>
                <p class="product-price">₹799</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>

        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://m.media-amazon.com/images/I/61v+W8M2pXL.SX522.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Allen Solly</p>
                <h3 class="product-name">Printed Cotton Shirt</h3>
                <p class="product-price">₹1,299</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
        
        <div class="product-card">
            <div class="product-img-container">
                <span class="product-badge">Exclusive</span>
                <img src="https://m.media-amazon.com/images/I/612wktfQEEL.SX522.jpg" alt="Product" class="product-img">
                <div class="product-overlay">
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                        <i class="fas fa-certificate" style="color: var(--luxury-gold);"></i> VIP Only
                    </p>
                </div>
            </div>
            <div class="product-info">
                <p class="product-brand">Peter England</p>
                <h3 class="product-name">Formal Cotton Shirt</h3>
                <p class="product-price">₹1,199</p>
                <div class="product-actions">
                    <button class="btn-luxury"><i class="fas fa-heart"></i> Save</button>
                    <button class="btn-luxury"><i class="fas fa-shopping-cart"></i> Add to Cart</button>
                </div>
            </div>
        </div>
    </div>
</section>
    
   
    
    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>VOGUE AI</h3>
                    <p>Revolutionizing luxury fashion discovery through artificial intelligence. Experience the future of personalized style curation.</p>
                    <div class="social-links">
                        <a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
                        <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a>
                        <a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a>
                        <a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
                    </div>
                </div>
                <div class="footer-section">
                    <h3>Features</h3>
                    <ul>
                        <li><a href="#">AI Personal Stylist</a></li>
                        <li><a href="#">Visual Search Engine</a></li>
                        <li><a href="#">Trend Forecasting</a></li>
                        <li><a href="#">Designer Collections</a></li>
                        <li><a href="#">VIP Access</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h3>Company</h3>
                    <ul>
                        <li><a href="#">About Us</a></li>
                        <li><a href="#">Careers</a></li>
                        <li><a href="#">Press</a></li>
                        <li><a href="#">Partnerships</a></li>
                        <li><a href="#">Contact</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h3>Support</h3>
                    <ul>
                        <li><a href="#">Help Center</a></li>
                        <li><a href="#">Privacy Policy</a></li>
                        <li><a href="#">Terms of Service</a></li>
                        <li><a href="#">Cookie Policy</a></li>
                        <li><a href="#">AI Guidelines</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2025 VOGUE AI. All rights reserved. ByteSquad, CDGI</p>
            </div>
        </div>
    </footer>
    
    <!-- Bootstrap Bundle -->
  <script>
  window.chatbaseConfig = {
    chatbotId: "mShJornVVOZmSPe-7RpMi", // your bot ID
    domain: "www.chatbase.co",

    chatButton: {
      color: "#d4af37",    // gold button
      textColor: "#000000",
      icon: "💬"
    },

    theme: {
      chatWindowBackground: "#0a0a0a",   // dark chat window
      botMessageBackground: "#1a1a1a",   // bot bubble
      userMessageBackground: "#d4af37",  // user bubble
      botTextColor: "#ffffff",
      userTextColor: "#000000",
      fontFamily: "Montserrat, sans-serif"
    }
  };
</script>

<script>
(function(){
  if(!window.chatbase||window.chatbase("getState")!=="initialized"){
    window.chatbase=(...arguments)=>{
      if(!window.chatbase.q){window.chatbase.q=[]}
      window.chatbase.q.push(arguments)
    };
    window.chatbase=new Proxy(window.chatbase,{
      get(target,prop){
        if(prop==="q"){return target.q}
        return(...args)=>target(prop,...args)
      }
    })
  }
  const onLoad=function(){
    const script=document.createElement("script");
    script.src="https://www.chatbase.co/embed.min.js";
    script.id="mShJornVVOZmSPe-7RpMi";  // your bot ID
    document.body.appendChild(script);
  };
  if(document.readyState==="complete"){onLoad()}
  else{window.addEventListener("load",onLoad)}
})();
</script>
</body>
</html>