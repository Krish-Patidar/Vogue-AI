package com.fashionauth;

public class Product {
    private String name;
    private String imageUrl;
    private double price;

    // AI analysis attributes
    private String style; // casual, formal, party
    private String colorMatch; // good, average, poor
    private double trendScore; // 0 to 10
    private String occasion; // work, party, casual
    private String verdict; // AI recommendation

    public Product(String name, String imageUrl, double price) {
        this.name = name;
        this.imageUrl = imageUrl;
        this.price = price;
    }

    // getters and setters
    public String getName() { return name; }
    public String getImageUrl() { return imageUrl; }
    public double getPrice() { return price; }
    public String getStyle() { return style; }
    public void setStyle(String style) { this.style = style; }
    public String getColorMatch() { return colorMatch; }
    public void setColorMatch(String colorMatch) { this.colorMatch = colorMatch; }
    public double getTrendScore() { return trendScore; }
    public void setTrendScore(double trendScore) { this.trendScore = trendScore; }
    public String getOccasion() { return occasion; }
    public void setOccasion(String occasion) { this.occasion = occasion; }
    public String getVerdict() { return verdict; }
    public void setVerdict(String verdict) { this.verdict = verdict; }
}
