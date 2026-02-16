import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'dart:math';

class Place {
  final String name;
  final String address;
  final LatLng location;
  final double? rating;
  final int? userRatingCount;
  final bool? isOpen;
  final String? phoneNumber;
  final String? websiteUri;
  final String? googleMapsUri;
  final String? photoReference;
  final List<String>? weekdayDescriptions;
  final double distance;

  Place({
    required this.name,
    required this.address,
    required this.location,
    this.rating,
    this.userRatingCount,
    this.isOpen,
    this.phoneNumber,
    this.websiteUri,
    this.googleMapsUri,
    this.photoReference,
    this.weekdayDescriptions,
    this.distance = 0,
  });

  factory Place.fromJson(Map<String, dynamic> json, LatLng center) {
    final loc = json['location'];
    final latLng = LatLng(
      (loc['latitude'] as num).toDouble(),
      (loc['longitude'] as num).toDouble(),
    );

    String? photoRef;
    final photos = json['photos'] as List?;
    if (photos != null && photos.isNotEmpty) {
      photoRef = photos[0]['name'] as String?;
    }

    final currentHours = json['currentOpeningHours'] as Map<String, dynamic>?;
    final regularHours = json['regularOpeningHours'] as Map<String, dynamic>?;

    bool? isOpen;
    if (currentHours != null && currentHours.containsKey('openNow')) {
      isOpen = currentHours['openNow'] as bool?;
    } else if (regularHours != null && regularHours.containsKey('openNow')) {
      isOpen = regularHours['openNow'] as bool?;
    }

    List<String>? weekdays;
    final weekdayDesc = (currentHours?['weekdayDescriptions'] ??
        regularHours?['weekdayDescriptions']) as List?;
    if (weekdayDesc != null) {
      weekdays = weekdayDesc.map((e) => e.toString()).toList();
    }

    return Place(
      name: (json['displayName'] as Map?)?['text'] ?? '',
      address: json['formattedAddress'] ?? '',
      location: latLng,
      rating: (json['rating'] as num?)?.toDouble(),
      userRatingCount: json['userRatingCount'] as int?,
      isOpen: isOpen,
      phoneNumber: json['nationalPhoneNumber'] as String?,
      websiteUri: json['websiteUri'] as String?,
      googleMapsUri: json['googleMapsUri'] as String?,
      photoReference: photoRef,
      weekdayDescriptions: weekdays,
      distance: _haversine(center, latLng),
    );
  }

  String? get photoUrl {
    if (photoReference == null) return null;
    return 'https://places.googleapis.com/v1/$photoReference/media?maxHeightPx=400&maxWidthPx=400&key=AIzaSyD_h40QWZPiY5biRspElBQuV2siVFW4IK4';
  }

  static double _haversine(LatLng p1, LatLng p2) {
    const R = 6371000.0;
    final dLat = (p2.latitude - p1.latitude) * pi / 180;
    final dLng = (p2.longitude - p1.longitude) * pi / 180;
    final a = sin(dLat / 2) * sin(dLat / 2) +
        cos(p1.latitude * pi / 180) *
            cos(p2.latitude * pi / 180) *
            sin(dLng / 2) *
            sin(dLng / 2);
    return R * 2 * atan2(sqrt(a), sqrt(1 - a));
  }
}
