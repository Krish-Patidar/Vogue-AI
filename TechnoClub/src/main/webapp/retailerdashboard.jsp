<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="jakarta.servlet.http.*,jakarta.servlet.*" %>
<%
    HttpSession session1 = request.getSession(false);
    if (session1 == null || session1.getAttribute("retailerId") == null) {
        response.sendRedirect("retailerlogin.jsp");
        return;
    }
    String retailerName = (String) session1.getAttribute("retailerName");
    String retailerEmail = (String) session1.getAttribute("retailerEmail");
%>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Retailer Dashboard | VOGUE AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

    <style>
        body {
            background: linear-gradient(135deg, #1c1c1c, #302b63, #0f0f0f);
            min-height: 100vh;
            font-family: 'Segoe UI', sans-serif;
            color: #fff;
        }
        .luxury-card {
            background: rgba(10, 10, 10, 0.9);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 15px;
            padding: 2rem;
            max-width: 650px;
            margin: 120px auto 40px auto;
            color: #fff;
            box-shadow: 0 0 25px rgba(0,0,0,0.5);
        }
        .luxury-title {
            color: #d4af37;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .form-control {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(212,175,55,0.3);
            color: #fff;
        }
        .form-control:focus {
            border-color: #d4af37;
            box-shadow: 0 0 10px rgba(212,175,55,0.5);
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        .btn-luxury {
            background: #d4af37;
            color: #000;
            font-weight: 600;
            border: none;
        }
        .btn-luxury:hover {
            background: #bfa335;
        }
        /* Navbar */
        .navbar {
            background: rgba(10, 10, 10, 0.8);
            border-bottom: 1px solid rgba(212, 175, 55, 0.3);
            backdrop-filter: blur(20px);
        }
        .navbar-brand {
            color: #d4af37 !important;
            font-weight: bold;
        }
    </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar navbar-expand-lg fixed-top">
    <div class="container">
        <a class="navbar-brand" href="#">
            <i class="fas fa-crown"></i> VOGUE AI
        </a>
        <div class="d-flex ms-auto align-items-center">
            <span class="me-3 text-light">Welcome, <%= retailerName %></span>
            <a href="LogoutServlet" class="btn btn-danger">Logout</a>
        </div>
    </div>
</nav>

<!-- Luxury Card Form -->
<div class="luxury-card">
    <h2 class="luxury-title"><i class="fas fa-plus-circle"></i> Add New Item</h2>

    <form action="AddItemServlet" method="post" enctype="multipart/form-data">
        <div class="mb-3">
            <label class="form-label">Item Name</label>
            <input type="text" name="item_name" class="form-control" placeholder="Enter item name" required>
        </div>

        <div class="mb-3">
            <label class="form-label">Price (₹)</label>
            <input type="number" step="0.01" name="price" class="form-control" placeholder="Enter price" required>
        </div>

        <div class="mb-3">
            <label class="form-label">Description</label>
            <textarea name="description" class="form-control" rows="3" placeholder="Enter item description" required></textarea>
        </div>

        <div class="mb-3">
            <label class="form-label">Upload Image</label>
            <input type="file" name="item_image" class="form-control" accept="image/*" required>
        </div>

        <button type="submit" class="btn btn-luxury w-100">Add Item</button>
    </form>
</div>

</body>
</html>
