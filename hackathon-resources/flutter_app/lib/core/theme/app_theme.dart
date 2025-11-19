import 'package:flutter/material.dart';

class AppTheme {
  // Colors
  static const Color primaryIndigo = Color(0xFF00082D);
  static const Color primaryPurple = Color(0xFF005EAC);
  static const Color primaryPink = Color(0xFF3D006C);
  static const Color successGreen = Color(0xFF10b981);
  static const Color textGray = Color(0xFF64748b);
  
  // Gradient for buttons
  static const LinearGradient buttonGradient = LinearGradient(
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
    colors: [
      primaryPurple,
      primaryPink,
    ],
  );
  
  // Theme Data
  static ThemeData get lightTheme {
    return ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: primaryIndigo),
      useMaterial3: true,
      fontFamily: 'Arial',
    );
  }
}
