package com.fashionauth;

public class Item {
    private int id;
    private String itemName;
    private double price;
    private String description;
    private String imageUrl;

    // ✅ Add a default constructor
    public Item() {
    }

    // Parameterized constructor (optional)
    public Item(int id, String itemName, double price, String description, String imageUrl) {
        this.id = id;
        this.itemName = itemName;
        this.price = price;
        this.description = description;
        this.imageUrl = imageUrl;
    }

    // Getters and setters
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }

    public String getItemName() { return itemName; }
    public void setItemName(String itemName) { this.itemName = itemName; }

    public double getPrice() { return price; }
    public void setPrice(double price) { this.price = price; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
}
