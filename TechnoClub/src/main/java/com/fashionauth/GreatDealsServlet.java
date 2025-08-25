package com.fashionauth;

import com.fashionauth.Item;
import java.io.IOException;
import java.sql.*;
import java.util.*;
import jakarta.servlet.*;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;

@WebServlet("/GreatDealsServlet")
public class GreatDealsServlet extends HttpServlet {

    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String category = request.getParameter("category");
        int userId = 2; // For testing, replace with session user id

        List<Item> items = new ArrayList<>();

        if (category != null && !category.isEmpty()) {
            try {
                Class.forName("com.mysql.cj.jdbc.Driver");
                Connection conn = DriverManager.getConnection(
                        "jdbc:mysql://localhost:3306/fashioondb", "root", "krrish@001#400"); // update credentials

                String sql = "SELECT i.id, i.item_name, i.price, i.description, i.image_url " +
                        "FROM items i " +
                        "JOIN wishlist w ON i.id = w.item_id " +
                        "JOIN item_categories ic ON i.id = ic.item_id " +
                        "WHERE w.user_id = ? AND TRIM(ic.category) = ? " +
                        "ORDER BY i.price ASC";

                PreparedStatement pst = conn.prepareStatement(sql);
                pst.setInt(1, userId);
                pst.setString(2, category.trim());

                ResultSet rs = pst.executeQuery();
                while (rs.next()) {
                    Item item = new Item();
                    item.setId(rs.getInt("id"));
                    item.setItemName(rs.getString("item_name"));
                    item.setPrice(rs.getDouble("price"));
                    item.setDescription(rs.getString("description"));
                    item.setImageUrl(rs.getString("image_url"));
                    items.add(item);
                }

                conn.close();

            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        request.setAttribute("items", items);
        RequestDispatcher rd = request.getRequestDispatcher("greatDeals.jsp");
        rd.forward(request, response);
    }
}
