package com.fashionauth;

import jakarta.servlet.*;
import jakarta.servlet.http.*;
import jakarta.servlet.annotation.*;
import java.io.IOException;
import java.sql.*;

@WebServlet("/AddToCartServlet")
public class AddToCartServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        int itemId = Integer.parseInt(request.getParameter("itemId"));
        Integer userId = (Integer) request.getSession().getAttribute("userId");

        if(userId == null){
            response.getWriter().write("fail");
            return;
        }

        try (Connection conn = DBConnection.getConnection()) {
            // Check if item already in cart
            PreparedStatement check = conn.prepareStatement(
                    "SELECT * FROM cart WHERE user_id=? AND item_id=?");
            check.setInt(1, userId);
            check.setInt(2, itemId);
            ResultSet rs = check.executeQuery();
            if (rs.next()) {
                response.getWriter().write("exists");
                return;
            }

            PreparedStatement ps = conn.prepareStatement(
                    "INSERT INTO cart(user_id, item_id) VALUES(?, ?)");
            ps.setInt(1, userId);
            ps.setInt(2, itemId);
            int rows = ps.executeUpdate();
            response.getWriter().write(rows > 0 ? "success" : "fail");
        } catch (SQLException e) {
            e.printStackTrace();
            response.getWriter().write("fail");
        }
    }
}
