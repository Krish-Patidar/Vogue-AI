package com.fashionauth;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;

@WebServlet("/UpdateUserServlet")
public class UpdateUserServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        HttpSession session = request.getSession(false);
        if (session == null || session.getAttribute("userId") == null) {
            response.sendRedirect("login.jsp");
            return;
        }

        int userId = (int) session.getAttribute("userId");
        String newName = request.getParameter("name");
        String newEmail = request.getParameter("email");

        try {
            Connection con = DBConnection.getConnection();
            PreparedStatement ps = con.prepareStatement(
                "UPDATE users SET name=?, email=? WHERE id=?"
            );
            ps.setString(1, newName);
            ps.setString(2, newEmail);
            ps.setInt(3, userId);

            int updated = ps.executeUpdate();

            if (updated > 0) {
                // Update session values
                session.setAttribute("userName", newName);
                session.setAttribute("userEmail", newEmail);
                response.sendRedirect("account.jsp?success=1");
            } else {
                response.sendRedirect("account.jsp?error=1");
            }

            ps.close();
            con.close();

        } catch (Exception e) {
            e.printStackTrace();
            response.sendRedirect("account.jsp?error=1");
        }
    }
}
