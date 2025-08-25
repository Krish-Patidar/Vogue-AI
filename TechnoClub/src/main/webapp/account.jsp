<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
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
  <title>My Account | VOGUE AI</title>

  <!-- Bootstrap & Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
   <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts - Minimalist -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700;900&family=Bebas+Neue&display=swap" rel="stylesheet">
    
  <style>
    :root {
      --luxury-gold: #d4af37;
      --luxury-bg: linear-gradient(135deg, #0f0f0f, #302B63);
    }
    body {
      background: var(--luxury-bg);
      color: #fff;
      font-family: 'Poppins', sans-serif;
      margin: 0;
    }
    /* Brand Header */
    .brand-header {
      padding: 1rem;
      text-align: center;
      background: rgba(0,0,0,0.6);
      border-bottom: 1px solid rgba(212,175,55,0.4);
    }
    .brand-header h1 {
      font-family: 'Cinzel', serif;
      font-size: 2rem;
      color: var(--luxury-gold);
      margin: 0;
      letter-spacing: 2px;
    }
    /* Container */
    .account-container {
      max-width: 1200px;
      margin: 2rem auto;
      background: rgba(20,20,20,0.9);
      border-radius: 20px;
      padding: 2rem;
      box-shadow: 0 0 40px rgba(0,0,0,0.6);
    }
    .profile-pic {
      width: 120px;
      height: 120px;
      border-radius: 50%;
      border: 3px solid var(--luxury-gold);
      object-fit: cover;
    }
    .sidebar {
      background: rgba(255,255,255,0.05);
      border-radius: 15px;
      padding: 1rem;
    }
    .nav-pills .nav-link {
      color: #fff;
      border-radius: 10px;
      margin: 0.4rem 0;
    }
    .nav-pills .nav-link.active {
      background: var(--luxury-gold);
      color: #000;
      font-weight: 600;
    }
    .order-card, .address-card {
      background: rgba(255,255,255,0.07);
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 1rem;
      border: 1px solid rgba(212,175,55,0.2);
    }
    h3 {
      color: var(--luxury-gold);
      font-weight: 600;
      margin-bottom: 1rem;
    }
    .btn-warning {
      background-color: var(--luxury-gold);
      border: none;
      color: #000;
    }
    .btn-outline-light:hover {
      background: var(--luxury-gold);
      color: #000;
    }
    @media (max-width: 768px) {
      .account-container {
        padding: 1rem;
      }
      .brand-header h1 {
        font-size: 1.5rem;
      }
    }
  </style>
</head>
<body>

  <!-- Brand Header -->
  <div class="brand-header">
    <h1>VOGUE AI</h1>
    <p class="text-muted">Your Personal Luxury Fashion Assistant</p>
  </div>

  <!-- Account Container -->
  <div class="container account-container">
    <div class="row g-4">
      
      <!-- Sidebar -->
      <div class="col-lg-3 col-md-4">
        <div class="sidebar text-center">
          <img src="./image/download.jpg" class="profile-pic mb-3" alt="Profile">
          <h4 class="mb-1"><%= userName %>
</h4>
          <p class="text-muted small">janedoe@email.com</p>
          <div class="nav flex-column nav-pills mt-3" id="account-tabs">
            <button class="nav-link active" data-bs-toggle="pill" data-bs-target="#orders"><i class="fas fa-shopping-bag me-2"></i> Orders</button>
            <button class="nav-link" data-bs-toggle="pill" data-bs-target="#details"><i class="fas fa-user me-2"></i> Personal Details</button>
            <button class="nav-link" data-bs-toggle="pill" data-bs-target="#address"><i class="fas fa-map-marker-alt me-2"></i> Addresses</button>
            <button class="nav-link" data-bs-toggle="pill" data-bs-target="#settings"><i class="fas fa-cog me-2"></i> Settings</button>
            <button class="nav-link text-danger"><a href="login.jsp"><i class="fas fa-sign-out-alt me-2"></i> Logout</a></button>
          </div>
        </div>
      </div>

      <!-- Content -->
      <div class="col-lg-9 col-md-8">
        <div class="tab-content">

          <!-- Orders -->
          <div class="tab-pane fade show active" id="orders">
            <h3><i class="fas fa-shopping-bag"></i> My Orders</h3>
            <div class="order-card">
              <h5>Order #12345</h5>
              <p>Gucci Leather Jacket — <span class="text-success">Delivered</span></p>
              <small>Ordered on: 12 Aug 2025</small>
            </div>
            <div class="order-card">
              <h5>Order #12346</h5>
              <p>Rolex Oyster Watch — <span class="text-warning">In Transit</span></p>
              <small>Ordered on: 15 Aug 2025</small>
            </div>
          </div>

       <!-- Personal Details -->
<div class="tab-pane fade" id="details">
  <h3><i class="fas fa-user"></i> Personal Details</h3>
  <form action="UpdateUserServlet" method="post">
    <div class="mb-3">
      <label class="form-label">Full Name</label>
      <input type="text" class="form-control" name="name"
             value="<%= userName %>" required>
    </div>
    <div class="mb-3">
      <label class="form-label">Email</label>
      <input type="email" class="form-control" name="email"
             value="<%= session.getAttribute("userEmail") %>" required>
    </div>
    <button type="submit" class="btn btn-warning">Update</button>
  </form>
</div>


          <!-- Addresses -->
          <div class="tab-pane fade" id="address">
            <h3><i class="fas fa-map-marker-alt"></i> Saved Addresses</h3>
            <div class="address-card">
              <h6>Home</h6>
              <p>123 Luxury Street, Indore, MP</p>
              <button class="btn btn-sm btn-outline-light">Edit</button>
            </div>
            <div class="address-card">
              <h6>Work</h6>
              <p>45 Corporate Park, Indore, MP</p>
              <button class="btn btn-sm btn-outline-light">Edit</button>
            </div>
          </div>

          <!-- Settings -->
          <div class="tab-pane fade" id="settings">
            <h3><i class="fas fa-cog"></i> Account Settings</h3>
            <p>Manage your account preferences here.</p>
            <button class="btn btn-outline-light">Change Password</button>
            <button class="btn btn-outline-light">Notification Preferences</button>
          </div>

        </div>
      </div>

    </div>
  </div>

  <!-- Bootstrap -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>