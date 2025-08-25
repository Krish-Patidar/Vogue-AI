<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ page import="java.sql.*, com.fashionauth.DBConnection" %>
<%
    Integer userId = (Integer) session.getAttribute("userId");
    int totalItems = 0;
    double totalValue = 0;

    Connection conn = null;
    PreparedStatement psTotal = null;
    PreparedStatement psItems = null;
    ResultSet rsTotal = null;
    ResultSet rsItems = null;

    if (userId != null) {
        try {
            conn = DBConnection.getConnection();

            // Query total items and total value
            psTotal = conn.prepareStatement(
                "SELECT SUM(c.quantity) as total_items, SUM(c.quantity * i.price) as total_value " +
                "FROM cart c JOIN items i ON c.item_id = i.id WHERE c.user_id=?"
            );
            psTotal.setInt(1, userId);
            rsTotal = psTotal.executeQuery();
            if (rsTotal.next()) {
                totalItems = rsTotal.getInt("total_items");
                totalValue = rsTotal.getDouble("total_value");
            }

            // Query all cart items
            psItems = conn.prepareStatement(
                "SELECT c.id AS cart_id, c.quantity, i.id AS item_id, i.item_name, i.price, i.image_url " +
                "FROM cart c JOIN items i ON c.item_id = i.id WHERE c.user_id=?"
            );
            psItems.setInt(1, userId);
            rsItems = psItems.executeQuery();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
%>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VOGUE AI - My Cart</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Montserrat:wght@300;400;500;600;700&family=Bebas+Neue&display=swap" rel="stylesheet">
    <!-- Bootstrap & Fonts & CSS (keep your CSS as is) -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Montserrat:wght@300;400;500;600;700&family=Bebas+Neue&display=swap" rel="stylesheet">
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
        
        /* Hero Section for Wishlist */
        .wishlist-header {
            padding: 120px 0 60px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .wishlist-title {
            font-family: 'Playfair Display', serif;
            font-size: 4rem;
            font-weight: 900;
            letter-spacing: -2px;
            line-height: 1.1;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #fff 0%, var(--luxury-gold) 50%, #fff 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 4s linear infinite, slideInDown 1s ease;
        }
        
        .wishlist-subtitle {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 300;
            letter-spacing: 2px;
            text-transform: uppercase;
            animation: fadeIn 1s ease 0.5s both;
            margin-bottom: 40px;
        }

        .wishlist-stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 30px;
            flex-wrap: wrap;
        }

        .wishlist-stat {
            text-align: center;
            padding: 20px;
        }

        .wishlist-stat-number {
            font-family: 'Bebas Neue', cursive;
            font-size: 2.5rem;
            color: var(--luxury-gold);
            display: block;
            line-height: 1;
        }

        .wishlist-stat-label {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
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
        
        /* Product Cards - Enhanced for Wishlist */
        .wishlist-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 35px;
            padding: 60px 0;
            max-width: 1400px;
            margin: 0 auto;
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

        .wishlist-heart {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(212, 175, 55, 0.5);
            color: var(--luxury-gold);
            padding: 10px;
            border-radius: 50%;
            font-size: 1.2rem;
            animation: heartPulse 2s infinite;
        }

        @keyframes heartPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
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
        
        .btn-remove {
            flex: 1;
            padding: 12px;
            background: transparent;
            border: 1px solid #ff4444;
            color: #ff4444;
            border-radius: 50px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .btn-remove:hover {
            background: #ff4444;
            color: #fff;
            transform: translateY(-2px);
        }

        .btn-add-to-cart {
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
        
        .btn-add-to-cart:hover {
            background: var(--luxury-gold);
            color: var(--primary-black);
            transform: translateY(-2px);
        }

        /* Empty Wishlist State */
        .empty-wishlist {
            text-align: center;
            padding: 100px 20px;
            max-width: 600px;
            margin: 0 auto;
        }

        .empty-wishlist-icon {
            font-size: 5rem;
            color: var(--luxury-gold);
            margin-bottom: 30px;
            opacity: 0.7;
        }

        .empty-wishlist-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            color: #fff;
            margin-bottom: 20px;
            font-weight: 700;
        }

        .empty-wishlist-text {
            color: rgba(255, 255, 255, 0.6);
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 40px;
        }

        .btn-discover {
            padding: 15px 40px;
            background: var(--gradient-gold);
            color: var(--primary-black);
            border: none;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }

        .btn-discover:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(212, 175, 55, 0.6);
            color: var(--primary-black);
            text-decoration: none;
        }
        
        /* Section Title */
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
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
            .wishlist-grid {
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 30px;
            }
        }

        @media (max-width: 768px) {
            .wishlist-title {
                font-size: 2.5rem;
            }
            
            .navbar-brand {
                font-size: 1.8rem;
            }
            
            .wishlist-grid {
                grid-template-columns: 1fr;
                gap: 25px;
                padding: 40px 0;
            }

            .wishlist-stats {
                gap: 20px;
            }

            .wishlist-stat {
                padding: 15px;
            }

            .wishlist-stat-number {
                font-size: 2rem;
            }
        }
        .wishlist-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr); /* 3 cards per row */
    gap: 35px;
    padding: 60px 0;
    max-width: 1400px;
    margin: 0 auto;
}


        
        .product-actions { display: flex; gap: 10px; margin-top: 20px; }
        .btn-remove, .btn-quantity { cursor: pointer; }
        .cart-stats { display: flex; justify-content: center; gap: 40px; margin-top: 30px; flex-wrap: wrap; text-align: center;}
        .wishlist-stat-number { font-family: 'Bebas Neue', cursive; font-size: 2.5rem; color: #d4af37; }
        .wishlist-stat-label { font-size: 0.8rem; color: rgba(255,255,255,0.6); text-transform: uppercase; margin-top: 5px;}
    </style></head>
<body>
<div class="luxury-bg"></div>

<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark fixed-top">
    <div class="container">
        <a class="navbar-brand" href="main.jsp"><i class="fas fa-crown"></i> VOGUE AI</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item"><a class="nav-link" href="dashboard.jsp">Home</a></li>
                <li class="nav-item"><a class="nav-link" href="discover.jsp">Discover</a></li>
                <li class="nav-item"><a class="nav-link" href="https://ai-wardrobe-advisor.vercel.app/">Compare</a></li>
                <li class="nav-item"><a class="nav-link" href="wishlist.jsp">Wishlist</a></li>
                <li class="nav-item"><a class="nav-link" href="cart.jsp"><i class="fas fa-shopping-cart"></i> Cart</a></li>
                <li class="nav-item"><a class="nav-link user-profile" href="account.jsp"><img src="./image/download.jpg" alt="User" class="user-avatar"></a></li>
            </ul>
        </div>
    </div>
</nav>

<!-- Cart Header -->
<section class="wishlist-header">
    <div class="container">
        <p class="wishlist-subtitle">Your Selected Items</p>
        <h1 class="wishlist-title">Shopping<br>Cart</h1>
        <div class="cart-stats">
            <div class="wishlist-stat">
                <span class="wishlist-stat-number" id="total-items"><%= totalItems %></span>
                <div class="wishlist-stat-label">Items in Cart</div>
            </div>
            <div class="wishlist-stat">
                <span class="wishlist-stat-number" id="total-value">₹<%= String.format("%.2f", totalValue) %></span>
                <div class="wishlist-stat-label">Total Value</div>
            </div>
        </div>
    </div>
</section>

<!-- Cart Items -->
<section class="container my-5">
    <div class="wishlist-grid">
        <%
            boolean hasItems = false;
            if (rsItems != null) {
                while(rsItems.next()) {
                    hasItems = true;
        %>
        <div class="product-card" data-cartid="<%= rsItems.getInt("cart_id") %>">
            <div class="product-img-container">
                <img src="<%= rsItems.getString("image_url") %>" width="80" alt="Product" class="product-img">
            </div>
            <div class="product-info">
                <h3 class="product-name"><%= rsItems.getString("item_name") %></h3>
                <p class="product-price"> ₹<%= rsItems.getDouble("price") %></p>
                <div class="product-actions">
                    <form action="BuyNowServlet" method="post" style="flex:1;">
                        <input type="hidden" name="cartId" value="<%= rsItems.getInt("cart_id") %>">
                        <button type="submit" class="btn-add-to-cart"> Buy Now</button>
                    </form>

                    

                    <button class="btn-remove" data-cartid="<%= rsItems.getInt("cart_id") %>">
                        <i class="fas fa-trash"></i> Remove
                    </button>
                </div>
            </div>
        </div>
        <%
                }
                rsItems.close();
            }
            if(rsTotal != null) rsTotal.close();
            if(psItems != null) psItems.close();
            if(psTotal != null) psTotal.close();
            if(conn != null) conn.close();

            if (!hasItems) {
        %>
        <div class="empty-wishlist">
            <i class="fas fa-shopping-cart empty-wishlist-icon"></i>
            <h2 class="empty-wishlist-title">Your Cart is Empty</h2>
            <p class="empty-wishlist-text">Start adding your favorite items!</p>
            <a href="discover.jsp" class="btn-discover">Shop Now</a>
        </div>
        <%
            }
        %>
    </div>
</section>

<script>
document.addEventListener('DOMContentLoaded', function() {

    function updateCartStats() {
        const cards = document.querySelectorAll('.product-card');
        let totalItems = 0, totalValue = 0;

        cards.forEach(card => {
            const qtyEl = card.querySelector('.quantity');
            const priceEl = card.querySelector('.product-price');

            let qty = parseInt(qtyEl?.innerText) || 1;
            let price = parseFloat(priceEl?.innerText.replace(/[^0-9.]/g, '')) || 0;

            totalItems += qty;
            totalValue += price * qty;
        });

        document.getElementById('total-items').innerText = totalItems;
        document.getElementById('total-value').innerText = `₹${totalValue.toLocaleString('en-IN')}`;
    }

    // Remove item
    document.querySelectorAll('.btn-remove').forEach(btn => {
        btn.addEventListener('click', function() {
            const card = this.closest('.product-card');
            const cartId = card.dataset.cartid;

            fetch('<%=request.getContextPath()%>/RemoveCartServlet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'cartId=' + cartId
            })
            .then(res => res.text())
            .then(data => {
                if(data.trim() === 'success'){
                    card.remove();
                    updateCartStats();
                    alert('Item removed from cart!');
                } else {
                    alert('Failed to remove item.');
                }
            });
        });
    });

    // Quantity increase/decrease
    document.querySelectorAll('.btn-quantity').forEach(btn => {
        btn.addEventListener('click', function() {
            const card = this.closest('.product-card');
            const cartId = card.dataset.cartid;
            const qtyEl = card.querySelector('.quantity');
            let qty = parseInt(qtyEl.innerText);

            const action = this.dataset.action;
            if(action === 'increase') qty++;
            else if(action === 'decrease' && qty > 1) qty--;

            // Update quantity via servlet
            fetch('<%=request.getContextPath()%>/UpdateCartServlet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: 'cartId=' + cartId + '&quantity=' + qty
            })
            .then(res => res.text())
            .then(data => {
                if(data.trim() === 'success') {
                    qtyEl.innerText = qty;
                    updateCartStats();
                } else {
                    alert('Failed to update quantity.');
                }
            });
        });
    });
});
</script>
</body>
</html>
