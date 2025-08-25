<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VOGUE AI - Premium Plans</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700;900&family=Bebas+Neue&display=swap" rel="stylesheet">
    
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
            --gradient-premium: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: #000;
            min-height: 100vh;
            color: #fff;
            overflow-x: hidden;
            font-weight: 400;
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
        
        /* Header Section */
        .premium-header {
            padding: 120px 0 60px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .premium-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 700;
            letter-spacing: -1px;
            line-height: 1.2;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #fff 0%, var(--luxury-gold) 50%, #fff 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 4s linear infinite;
        }
        
        .premium-subtitle {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 300;
            max-width: 600px;
            margin: 0 auto 40px;
            line-height: 1.6;
        }

        /* Plan Type Toggle */
        .plan-toggle {
            display: flex;
            justify-content: center;
            margin: 40px 0;
        }

        .toggle-container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(212, 175, 55, 0.2);
            border-radius: 50px;
            padding: 4px;
            display: flex;
            backdrop-filter: blur(10px);
        }

        .toggle-btn {
            padding: 12px 30px;
            background: transparent;
            border: none;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
            font-weight: 500;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .toggle-btn.active {
            background: var(--luxury-gold);
            color: var(--primary-black);
            font-weight: 600;
        }
        
        /* Pricing Cards */
        .pricing-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 20px;
        }
        
        .pricing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }
        
        .pricing-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px 30px;
            position: relative;
            transition: all 0.4s ease;
            overflow: hidden;
        }

        .pricing-card.popular {
            border-color: var(--luxury-gold);
            box-shadow: 0 20px 60px rgba(212, 175, 55, 0.2);
            transform: scale(1.05);
        }

        .pricing-card.popular::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(212, 175, 55, 0.05) 100%);
            pointer-events: none;
        }
        
        .pricing-card:hover {
            transform: translateY(-10px) scale(1.02);
            border-color: var(--luxury-gold);
            box-shadow: 0 30px 70px rgba(212, 175, 55, 0.3);
        }

        .popular-badge {
            position: absolute;
            top: -1px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--gradient-gold);
            color: var(--primary-black);
            padding: 8px 25px;
            border-radius: 0 0 20px 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .plan-name {
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 10px;
            text-align: center;
        }

        .plan-description {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 30px;
            line-height: 1.5;
        }
        
        .price-container {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .price {
            font-size: 3rem;
            font-weight: 300;
            color: var(--luxury-gold);
            font-family: 'Inter', sans-serif;
            display: flex;
            align-items: baseline;
            justify-content: center;
            gap: 5px;
        }

        .price .currency {
            font-size: 1.5rem;
        }

        .price .period {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.6);
            font-weight: 400;
        }

        .free-text {
            font-size: 2.5rem;
            font-weight: 500;
            color: var(--luxury-gold);
        }
        
        .features-list {
            list-style: none;
            padding: 0;
            margin: 30px 0;
        }
        
        .features-list li {
            padding: 12px 0;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            line-height: 1.5;
        }
        
        .features-list li i {
            color: var(--luxury-gold);
            font-size: 1rem;
            margin-top: 2px;
            flex-shrink: 0;
        }

        .feature-unavailable {
            color: rgba(255, 255, 255, 0.3) !important;
        }

        .feature-unavailable i {
            color: rgba(255, 255, 255, 0.2) !important;
        }
        
        .cta-button {
            width: 100%;
            padding: 15px;
            background: transparent;
            border: 1px solid rgba(212, 175, 55, 0.5);
            color: var(--luxury-gold);
            border-radius: 12px;
            font-size: 0.95rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .cta-button.premium {
            background: var(--gradient-gold);
            color: var(--primary-black);
            border-color: transparent;
            font-weight: 600;
        }
        
        .cta-button:hover {
            background: var(--luxury-gold);
            color: var(--primary-black);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(212, 175, 55, 0.4);
        }

        .cta-button.premium:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(212, 175, 55, 0.6);
        }

        /* Features Comparison */
        .features-comparison {
            max-width: 1000px;
            margin: 80px auto;
            padding: 0 20px;
        }

        .comparison-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            color: #fff;
            margin-bottom: 50px;
        }

        .comparison-table {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            overflow: hidden;
        }

        .comparison-row {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .comparison-row:last-child {
            border-bottom: none;
        }

        .comparison-row.header {
            background: rgba(212, 175, 55, 0.1);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.85rem;
        }

        .comparison-feature {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }

        .comparison-check {
            text-align: center;
            font-size: 1.2rem;
        }

        .check-yes {
            color: var(--luxury-gold);
        }

        .check-no {
            color: rgba(255, 255, 255, 0.3);
        }

        .check-limited {
            color: #ffa500;
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
        
        /* Responsive Design */
        @media (max-width: 1200px) {
            .pricing-grid {
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }

            .pricing-card.popular {
                transform: none;
            }
        }

        @media (max-width: 768px) {
            .premium-title {
                font-size: 2.5rem;
            }
            
            .navbar-brand {
                font-size: 1.8rem;
            }
            
            .pricing-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }

            .pricing-card {
                padding: 30px 20px;
            }

            .comparison-row {
                grid-template-columns: 1fr;
                gap: 10px;
                text-align: center;
                padding: 15px;
            }

            .comparison-row.header {
                display: none;
            }

            .toggle-container {
                flex-direction: column;
                width: 100%;
                max-width: 300px;
                margin: 0 auto;
            }

            .toggle-btn {
                width: 100%;
                margin: 2px 0;
            }
        }
    </style>
</head>
<body>
   <!-- Luxury Background -->
  <div class="luxury-bg"></div>
  <div class="particles" id="particles"></div>
  
   <!-- Navigation -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top">
    <div class="container">
        <a class="navbar-brand" href="#">
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
    

    <!-- Premium Header -->
    <section class="premium-header">
        <div class="container">
            <h1 class="premium-title">Unlock Premium Experience</h1>
            <p class="premium-subtitle">Choose the perfect plan to elevate your fashion discovery journey with AI-powered features and exclusive access to luxury collections.</p>
            
            <!-- Plan Type Toggle -->
            <div class="plan-toggle">
                <div class="toggle-container">
                    <button class="toggle-btn active" data-type="users">For Users</button>
                    <button class="toggle-btn" data-type="retailers">For Retailers</button>
                </div>
            </div>
        </div>
    </section>

    <!-- Pricing Plans for Users -->
    <section class="pricing-container" id="users-plans">
        <div class="pricing-grid">
            <!-- Free Plan -->
            <div class="pricing-card">
                <div class="plan-name">Free Plan</div>
                <div class="plan-description">Essential features for fashion discovery</div>
                <div class="price-container">
                    <div class="free-text">Free</div>
                </div>
                <ul class="features-list">
                    <li><i class="fas fa-check"></i> Browse products</li>
                    <li><i class="fas fa-check"></i> Add to wishlist</li>
                    <li><i class="fas fa-check"></i> Add to cart</li>
                    <li><i class="fas fa-check"></i> Basic search functionality</li>
                    <li><i class="fas fa-check"></i> View featured products</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> AI-powered wardrobe suggestions</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Compare features</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Visual search</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Personal stylist recommendations</li>
                </ul>
                <button class="cta-button">Get Started</button>
            </div>

            <!-- Premium Plan -->
            <div class="pricing-card popular">
                <div class="popular-badge">Most Popular</div>
                <div class="plan-name">Premium Plan</div>
                <div class="plan-description">AI-powered fashion with advanced features</div>
                <div class="price-container">
                    <div class="price">
                        <span class="currency">₹</span>
                        <span>899</span>
                        <span class="period">/month</span>
                    </div>
                </div>
                <ul class="features-list">
                    <li><i class="fas fa-check"></i> All Free Plan features</li>
                    <li><i class="fas fa-check"></i> AI-powered wardrobe suggestions</li>
                    <li><i class="fas fa-check"></i> Advanced compare features</li>
                    <li><i class="fas fa-check"></i> Visual search technology</li>
                    <li><i class="fas fa-check"></i> Trend forecasting</li>
                    <li><i class="fas fa-check"></i> Style matching algorithm</li>
                    <li><i class="fas fa-check"></i> Priority customer support</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Personal stylist consultation</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Early access to features</li>
                </ul>
                <button class="cta-button premium">Start Premium</button>
            </div>

            <!-- Premium+ Plan -->
            <div class="pricing-card">
                <div class="plan-name">Premium+</div>
                <div class="plan-description">Full AI suite with personal stylist access</div>
                <div class="price-container">
                    <div class="price">
                        <span class="currency">₹</span>
                        <span>1299</span>
                        <span class="period">/month</span>
                    </div>
                </div>
                <ul class="features-list">
                    <li><i class="fas fa-check"></i> All Premium features</li>
                    <li><i class="fas fa-check"></i> Full AI suite access</li>
                    <li><i class="fas fa-check"></i> Personal stylist recommendations</li>
                    <li><i class="fas fa-check"></i> 1-on-1 virtual styling sessions</li>
                    <li><i class="fas fa-check"></i> Early access to new features</li>
                    <li><i class="fas fa-check"></i> Exclusive designer previews</li>
                    <li><i class="fas fa-check"></i> VIP customer support</li>
                    <li><i class="fas fa-check"></i> Custom wardrobe planning</li>
                    <li><i class="fas fa-check"></i> Advanced analytics dashboard</li>
                </ul>
                <button class="cta-button">Go Premium+</button>
            </div>
        </div>
    </section>

    <!-- Pricing Plans for Retailers -->
    <section class="pricing-container" id="retailers-plans" style="display: none;">
        <div class="pricing-grid">
            <!-- Free Plan for Retailers -->
            <div class="pricing-card">
                <div class="plan-name">Basic</div>
                <div class="plan-description">Get started with limited product listings</div>
                <div class="price-container">
                    <div class="free-text">Free</div>
                </div>
                <ul class="features-list">
                    <li><i class="fas fa-check"></i> Limited product uploads (50/month)</li>
                    <li><i class="fas fa-check"></i> Basic product analytics</li>
                    <li><i class="fas fa-check"></i> Display basic ads</li>
                    <li><i class="fas fa-check"></i> Featured product promotion</li>
                    <li><i class="fas fa-check"></i> Standard listing visibility</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Unlimited uploads</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Advanced analytics</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Priority listing</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> API access</li>
                </ul>
                <button class="cta-button">Get Started</button>
            </div>

            <!-- Premium Plan for Retailers -->
            <div class="pricing-card popular">
                <div class="popular-badge">Most Popular</div>
                <div class="plan-name">Professional</div>
                <div class="plan-description">Unlimited uploads with advanced analytics</div>
                <div class="price-container">
                    <div class="price">
                        <span class="currency">₹</span>
                        <span>1999</span>
                        <span class="period">/month</span>
                    </div>
                </div>
                <ul class="features-list">
                    <li><i class="fas fa-check"></i> Unlimited product uploads</li>
                    <li><i class="fas fa-check"></i> Advanced product performance analytics</li>
                    <li><i class="fas fa-check"></i> Featured product listings</li>
                    <li><i class="fas fa-check"></i> Enhanced product visibility</li>
                    <li><i class="fas fa-check"></i> Customer insights dashboard</li>
                    <li><i class="fas fa-check"></i> Marketing campaign tools</li>
                    <li><i class="fas fa-check"></i> Priority customer support</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Bulk upload APIs</li>
                    <li class="feature-unavailable"><i class="fas fa-times"></i> Dedicated account manager</li>
                </ul>
                <button class="cta-button premium">Start Professional</button>
            </div>

            <!-- Enterprise Plan for Retailers -->
            <div class="pricing-card">
                <div class="plan-name">Enterprise</div>
                <div class="plan-description">Complete solution with priority support</div>
                <div class="price-container">
                    <div class="price">
                        <span class="currency">₹</span>
                        <span>4999</span>
                        <span class="period">/month</span>
                    </div>
                </div>
                <ul class="features-list">
                    <li><i class="fas fa-check"></i> All Professional features</li>
                    <li><i class="fas fa-check"></i> Priority listing placement</li>
                    <li><i class="fas fa-check"></i> Advanced analytics & reporting</li>
                    <li><i class="fas fa-check"></i> Bulk upload APIs</li>
                    <li><i class="fas fa-check"></i> Marketing campaign support</li>
                    <li><i class="fas fa-check"></i> Custom integration options</li>
                    <li><i class="fas fa-check"></i> Dedicated account manager</li>
                    <li><i class="fas fa-check"></i> White-label solutions</li>
                    <li><i class="fas fa-check"></i> 24/7 premium support</li>
                </ul>
                <button class="cta-button">Go Enterprise</button>
            </div>
        </div>
    </section>

    <!-- Features Comparison Table -->
    <section class="features-comparison">
        <h2 class="comparison-title">Compare All Features</h2>
        <div class="comparison-table" id="users-comparison">
            <div class="comparison-row header">
                <div>Features</div>
                <div>Free</div>
                <div>Premium</div>
                <div>Premium+</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Browse Products</div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Add to Wishlist & Cart</div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Basic Search</div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">AI Wardrobe Suggestions</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Visual Search</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Compare Features</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Personal Stylist</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Early Access to Features</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Customer Support</div>
                <div class="comparison-check check-limited">Basic</div>
                <div class="comparison-check check-yes">Priority</div>
                <div class="comparison-check check-yes">VIP</div>
            </div>
        </div>

        <div class="comparison-table" id="retailers-comparison" style="display: none;">
            <div class="comparison-row header">
                <div>Features</div>
                <div>Basic</div>
                <div>Professional</div>
                <div>Enterprise</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Product Uploads</div>
                <div class="comparison-check check-limited">50/month</div>
                <div class="comparison-check check-yes">Unlimited</div>
                <div class="comparison-check check-yes">Unlimited</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Product Analytics</div>
                <div class="comparison-check check-limited">Basic</div>
                <div class="comparison-check check-yes">Advanced</div>
                <div class="comparison-check check-yes">Premium</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Featured Listings</div>
                <div class="comparison-check check-limited">Limited</div>
                <div class="comparison-check check-yes"><i class="fas fa-check"></i></div>
                <div class="comparison-check check-yes">Priority</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">API Access</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes">Bulk Upload</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Marketing Support</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes">Tools</div>
                <div class="comparison-check check-yes">Full Support</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Account Manager</div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-no"><i class="fas fa-times"></i></div>
                <div class="comparison-check check-yes">Dedicated</div>
            </div>
            <div class="comparison-row">
                <div class="comparison-feature">Customer Support</div>
                <div class="comparison-check check-limited">Basic</div>
                <div class="comparison-check check-yes">Priority</div>
                <div class="comparison-check check-yes">24/7</div>
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
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        // Create floating particles
        function createParticles() {
            const particles = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {
                const particle = document.createElement('div');
                particle.classList.add('particle');
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 15 + 's';
                particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
                particles.appendChild(particle);
            }
        }

        // Plan type toggle functionality
        document.addEventListener('DOMContentLoaded', function() {
            createParticles();

            const toggleBtns = document.querySelectorAll('.toggle-btn');
            const usersPlans = document.getElementById('users-plans');
            const retailersPlans = document.getElementById('retailers-plans');
            const usersComparison = document.getElementById('users-comparison');
            const retailersComparison = document.getElementById('retailers-comparison');

            toggleBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    // Remove active class from all buttons
                    toggleBtns.forEach(b => b.classList.remove('active'));
                    // Add active class to clicked button
                    this.classList.add('active');

                    const type = this.dataset.type;
                    
                    if (type === 'users') {
                        usersPlans.style.display = 'block';
                        retailersPlans.style.display = 'none';
                        usersComparison.style.display = 'block';
                        retailersComparison.style.display = 'none';
                    } else {
                        usersPlans.style.display = 'none';
                        retailersPlans.style.display = 'block';
                        usersComparison.style.display = 'none';
                        retailersComparison.style.display = 'block';
                    }
                });
            });

            // CTA button interactions
            const ctaButtons = document.querySelectorAll('.cta-button');
            ctaButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const originalText = this.textContent;
                    this.textContent = 'Processing...';
                    this.style.opacity = '0.7';
                    
                    setTimeout(() => {
                        this.textContent = 'Success!';
                        this.style.background = '#4CAF50';
                        this.style.borderColor = '#4CAF50';
                        
                        setTimeout(() => {
                            this.textContent = originalText;
                            this.style.opacity = '1';
                            this.style.background = '';
                            this.style.borderColor = '';
                        }, 1500);
                    }, 1000);
                });
            });
        });

        // Navbar scroll effect
        window.addEventListener('scroll', function() {
            const navbar = document.querySelector('.navbar');
            if (window.scrollY > 50) {
                navbar.style.background = 'rgba(10, 10, 10, 0.95)';
            } else {
                navbar.style.background = 'rgba(10, 10, 10, 0.8)';
            }
        });

        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Add hover effects to pricing cards
        const pricingCards = document.querySelectorAll('.pricing-card');
        pricingCards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-15px) scale(1.02)';
            });
            
            card.addEventListener('mouseleave', function() {
                if (this.classList.contains('popular')) {
                    this.style.transform = 'scale(1.05)';
                } else {
                    this.style.transform = 'translateY(0) scale(1)';
                }
            });
        });
    </script>
</body>
</html>