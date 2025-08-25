package com.fashionauth;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@WebServlet("/RemoveCartServlet")
public class RemoveCartServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String cartIdStr = request.getParameter("cartId"); // get cartId from JS
        if(cartIdStr == null || cartIdStr.isEmpty()) {
            response.getWriter().write("fail");
            return;
        }

        try {
            int cartId = Integer.parseInt(cartIdStr);

            Connection conn = DBConnection.getConnection();
            PreparedStatement ps = conn.prepareStatement("DELETE FROM cart WHERE id=?");
            ps.setInt(1, cartId);
            int rowsAffected = ps.executeUpdate();

            ps.close();
            conn.close();

            if (rowsAffected > 0) {
                response.getWriter().write("success");
            } else {
                response.getWriter().write("fail");
            }
        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().write("fail");
        }
    }
}
