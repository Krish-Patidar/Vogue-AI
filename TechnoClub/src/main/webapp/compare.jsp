<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Fashion Comparison</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        .product-list { display: flex; gap: 20px; flex-wrap: wrap; }
        .product-card { border: 1px solid #ccc; border-radius: 8px; padding: 10px; width: 200px; text-align: center; background: #fff; }
        .product-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 8px; }
        .compare-section { margin-top: 30px; }
        #compare-container { display: flex; gap: 20px; flex-wrap: wrap; }
        .product-compare-card { border: 1px solid #ccc; border-radius: 8px; padding: 10px; width: 200px; text-align: center; background: #fff; }
        .ai-verdict { font-weight: bold; color: #d4af37; margin-top: 10px; }
        .ai-assistant-btn { position: fixed; bottom: 20px; right: 20px; background: #d4af37; color: #000; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    </style>
</head>
<body>

<h1>Fashion Products</h1>

<div class="product-list">
    <div class="product-card">
        <img src="images/red_dress.jpg" alt="Red Dress">
        <h3>Red Dress</h3>
        <p>Price: ₹1500</p>
        <button class="compare-btn" data-id="1">Add to Compare</button>
    </div>

    <div class="product-card">
        <img src="images/blue_blouse.jpg" alt="Blue Blouse">
        <h3>Blue Blouse</h3>
        <p>Price: ₹1200</p>
        <button class="compare-btn" data-id="2">Add to Compare</button>
    </div>

    <div class="product-card">
        <img src="images/green_shirt.jpg" alt="Green Shirt">
        <h3>Green Shirt</h3>
        <p>Price: ₹1000</p>
        <button class="compare-btn" data-id="3">Add to Compare</button>
    </div>
</div>

<div class="compare-section" id="compare-section">
    <h2>AI Fashion Comparison</h2>
    <p id="no-products-msg">No products selected for comparison.</p>
    <div id="compare-container"></div>
</div>

<div class="ai-assistant-btn" title="Ask AI Stylist">
    <i class="fas fa-robot"></i>
</div>

<script>
    // Correct product data mapping
    const productsData = {
        1: { name: "Red Dress", image: "images/red_dress.jpg", price: 1500, color: "Red", trendiness: 8, occasion: "Party" },
        2: { name: "Blue Blouse", image: "images/blue_blouse.jpg", price: 1200, color: "Blue", trendiness: 6, occasion: "Casual" },
        3: { name: "Green Shirt", image: "images/green_shirt.jpg", price: 1000, color: "Green", trendiness: 7, occasion: "Work" }
    };

    let selectedProducts = [];

    document.querySelectorAll('.compare-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = btn.getAttribute('data-id');
            if(!selectedProducts.includes(id)){
                selectedProducts.push(id);
            }
            renderComparison();
        });
    });

    function renderComparison() {
        const container = document.getElementById('compare-container');
        const msg = document.getElementById('no-products-msg');
        container.innerHTML = "";

        if(selectedProducts.length === 0){
            msg.style.display = "block";
            return;
        }
        msg.style.display = "none";

        selectedProducts.forEach(id => {
            const p = productsData[id];
            const verdict = aiFashionVerdict(p);

            const card = document.createElement('div');
            card.classList.add('product-compare-card');
            card.innerHTML = `
                <img src="${p.image}" alt="${p.name}">
                <h3>${p.name}</h3>
                <p>Price: ₹${p.price}</p>
                <p>Color: ${p.color}</p>
                <p class="ai-verdict">AI Verdict: ${verdict}</p>
            `;
            container.appendChild(card);
        });
    }

    function aiFashionVerdict(product){
        if(product.trendiness >= 8) return "Highly Trendy - Perfect for Parties";
        if(product.occasion === "Work") return "Suitable for Work";
        if(product.price < 1300) return "Affordable Casual Choice";
        return "Stylish Choice for Any Occasion";
    }

    document.querySelector('.ai-assistant-btn').addEventListener('click', () => {
        alert("Select products to compare and get AI styling tips!");
    });
</script>

</body>
</html>
