package com.fashionauth;

import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.MultipartConfig;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import jakarta.servlet.http.Part;

@WebServlet("/AddItemServlet")
@MultipartConfig(fileSizeThreshold = 1024 * 1024 * 2, // 2MB
                 maxFileSize = 1024 * 1024 * 10,      // 10MB
                 maxRequestSize = 1024 * 1024 * 50)   // 50MB
public class AddItemServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    // Change this to your project upload folder path
    private static final String UPLOAD_DIR = "uploads";

    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        HttpSession session = request.getSession(false);
        if (session == null || session.getAttribute("retailerId") == null) {
            response.sendRedirect("retailerlogin.jsp");
            return;
        }

        int retailerId = (int) session.getAttribute("retailerId");
        String itemName = request.getParameter("item_name");
        String priceStr = request.getParameter("price");
        String description = request.getParameter("description");

        // Handle image upload
        Part filePart = request.getPart("item_image");
        String fileName = System.currentTimeMillis() + "_" + filePart.getSubmittedFileName();

        String appPath = request.getServletContext().getRealPath("");
        String uploadPath = appPath + File.separator + UPLOAD_DIR;

        // Create uploads folder if not exists
        File uploadDir = new File(uploadPath);
        if (!uploadDir.exists()) uploadDir.mkdir();

        String filePath = uploadPath + File.separator + fileName;
        filePart.write(filePath);

        // Store relative path to DB (not absolute path)
        String imageUrl = UPLOAD_DIR + "/" + fileName;

        try {
            double price = Double.parseDouble(priceStr);

            Connection con = DBConnection.getConnection();
            PreparedStatement ps = con.prepareStatement(
                "INSERT INTO items (retailer_id, item_name, price, description, image_url) VALUES (?, ?, ?, ?, ?)"
            );
            ps.setInt(1, retailerId);
            ps.setString(2, itemName);
            ps.setDouble(3, price);
            ps.setString(4, description);
            ps.setString(5, imageUrl);

            ps.executeUpdate();

            ps.close();
            con.close();

            response.sendRedirect("retailerdashboard.jsp?success=1");

        } catch (Exception e) {
            e.printStackTrace();
            response.sendRedirect("retailerdashboard.jsp?error=1");
        }
    }
}
