package com.fashionauth;

import jakarta.servlet.*;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.util.*;

@WebServlet("/CompareServlet")
public class CompareServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        // Simulated products selected by user
        List<Product> products = new ArrayList<>();
        products.add(new Product("Red Dress", "images/red_dress.jpg", 1500));
        products.add(new Product("Blue Blouse", "images/blue_blouse.jpg", 1200));

        // Call AI analysis
        for(Product p : products) {
            performAIAnalysis(p);
        }

        request.setAttribute("products", products);
        RequestDispatcher rd = request.getRequestDispatcher("compare.jsp");
        rd.forward(request, response);
    }

    private void performAIAnalysis(Product p) {
        // This is placeholder AI logic. Replace with actual ML/AI model
        if(p.getName().toLowerCase().contains("dress")) {
            p.setStyle("Party");
            p.setOccasion("Party");
            p.setColorMatch("Good");
            p.setTrendScore(8.5);
            p.setVerdict("Best for evening parties and casual outings");
        } else {
            p.setStyle("Casual");
            p.setOccasion("Work");
            p.setColorMatch("Average");
            p.setTrendScore(7.0);
            p.setVerdict("Suitable for office and casual wear");
        }
    }
}
