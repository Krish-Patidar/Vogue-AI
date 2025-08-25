package com.fashionauth;

import java.io.IOException;
import java.sql.*;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;

@WebServlet("/RemoveWishlistServlet")
public class RemoveWishlistServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        HttpSession session = request.getSession();
        Integer userId = (Integer) session.getAttribute("userId");
        if(userId == null) {
            response.getWriter().write("failure");
            return;
        }

        String itemIdStr = request.getParameter("itemId");
        if(itemIdStr == null || itemIdStr.isEmpty()) {
            response.getWriter().write("failure");
            return;
        }

        int itemId = Integer.parseInt(itemIdStr);

        try (Connection conn = DBConnection.getConnection()) {
            String sql = "DELETE FROM wishlist WHERE user_id=? AND item_id=?";
            PreparedStatement ps = conn.prepareStatement(sql);
            ps.setInt(1, userId);
            ps.setInt(2, itemId);

            int rows = ps.executeUpdate();
            if(rows > 0){
                response.getWriter().write("success");
            } else {
                response.getWriter().write("failure");
            }
            ps.close();
        } catch (SQLException e) {
            e.printStackTrace();
            response.getWriter().write("failure");
        }
    }
}
