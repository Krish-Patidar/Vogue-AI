<%@ page contentType="text/html;charset=UTF-8" %>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login | VOGUE AI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts - Minimalist -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700;900&family=Bebas+Neue&display=swap" rel="stylesheet">
  
  <style>
    body {
      background: linear-gradient(135deg, #1c1c1c, #302b63, #0f0f0f);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      font-family: 'Segoe UI', sans-serif;
    }
    .auth-card {
      background: rgba(10, 10, 10, 0.9);
      border: 1px solid rgba(212, 175, 55, 0.3);
      border-radius: 15px;
      padding: 2rem;
      max-width: 400px;
      width: 100%;
      color: #fff;
      box-shadow: 0 0 25px rgba(0,0,0,0.5);
    }
    .auth-title {
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
    .auth-link {
      color: #d4af37;
      text-decoration: none;
    }
    .auth-link:hover {
      text-decoration: underline;
    }
     body {
      background: linear-gradient(135deg, #1c1c1c, #302b63, #0f0f0f);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      font-family: 'Segoe UI', sans-serif;
    }
    .auth-card {
      background: rgba(10, 10, 10, 0.9);
      border: 1px solid rgba(212, 175, 55, 0.3);
      border-radius: 15px;
      padding: 2rem;
      max-width: 400px;
      width: 100%;
      color: #fff;
      box-shadow: 0 0 25px rgba(0,0,0,0.5);
    }
    .auth-title {
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
    .auth-link {
      color: #d4af37;
      text-decoration: none;
    }
    .auth-link:hover {
      text-decoration: underline;
    }
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
        .btn-auth {
    padding: 10px 25px;
    margin: 0 5px;
    border-radius: 50px;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    text-decoration: none;
    font-size: 0.85rem;
}

.btn-login {
    background: transparent;
    border: 1px solid var(--luxury-gold);
    color: var(--luxury-gold);
}

.btn-login:hover {
    background: var(--luxury-gold);
    color: var(--primary-black);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(212, 175, 55, 0.4);
}

.btn-register {
    background: var(--gradient-gold);
    border: none;
    color: var(--primary-black);
}

.btn-register:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(212, 175, 55, 0.6);
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
            <a class="navbar-brand" href="index.jsp">
                <i class="fas fa-crown"></i> VOGUE AI
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="btn-auth btn-login" href="login.jsp">Login</a>
                    </li>
                    <li class="nav-item">
                        <a class="btn-auth btn-login" href="register.jsp">Register</a>
                    </li>
                    <li class="nav-item">
                        <a class="btn-auth btn-register" href="retailerlogin.jsp">Retailer Login</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    
        <br><br>
  <div class="auth-card">
    <h2 class="auth-title"><i class="fas fa-user-circle"></i> Retailer Login</h2>
    <!-- Important: Use LoginServlet here -->
<form action="RetailerServlet" method="post">
      <div class="mb-3">
        <label for="email" class="form-label">Email address</label>
        <input type="email" id="email" name="email" class="form-control" placeholder="Enter your email" required>
      </div>
      <div class="mb-3">
        <label for="password" class="form-label">Password</label>
        <input type="password" id="password" name="password" class="form-control" placeholder="Enter your password" required>
      </div>
      <button type="submit" class="btn-register btn-luxury w-100">Login</button>
    </form>
  </div>
</body>
</html>
