package com.fashionauth;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;

@WebServlet("/AddWishlistServlet")
public class AddWishlistServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    protected void doPost(HttpServletRequest req, HttpServletResponse res)
            throws ServletException, IOException {
        HttpSession session = req.getSession(false);

        if (session == null || session.getAttribute("userId") == null) {
            res.sendRedirect("login.jsp");
            return;
        }

        int userId = (Integer) session.getAttribute("userId");
        String itemIdStr = req.getParameter("itemId");

        try {
            int itemId = Integer.parseInt(itemIdStr);

            Connection conn = DBConnection.getConnection();
            PreparedStatement ps = conn.prepareStatement(
                "INSERT INTO wishlist (user_id, item_id) VALUES (?, ?)"
            );
            ps.setInt(1, userId);
            ps.setInt(2, itemId);
            ps.executeUpdate();

            res.sendRedirect("wishlist.jsp?success=1");

        } catch (Exception e) {
            e.printStackTrace();
            res.sendRedirect("wishlist.jsp?error=1");
        }
    }
}
