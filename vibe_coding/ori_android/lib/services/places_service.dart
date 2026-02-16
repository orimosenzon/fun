import 'dart:convert';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:http/http.dart' as http;
import '../models/place.dart';
import 'gemini_service.dart';

class PlacesService {
  static const _apiKey = 'AIzaSyD_h40QWZPiY5biRspElBQuV2siVFW4IK4';

  static Future<List<Place>> searchPlaces({
    required ParsedQuery parsedQuery,
    required LatLng center,
    required double radius,
    required String lang,
  }) async {
    const fieldMask = 'places.displayName,places.formattedAddress,places.location,'
        'places.rating,places.userRatingCount,places.photos,'
        'places.regularOpeningHours,places.currentOpeningHours,'
        'places.priceLevel,places.nationalPhoneNumber,'
        'places.websiteUri,places.googleMapsUri';

    final body = <String, dynamic>{
      'textQuery': parsedQuery.textQuery,
      'maxResultCount': 20,
      'languageCode': lang,
      'locationBias': {
        'circle': {
          'center': {
            'latitude': center.latitude,
            'longitude': center.longitude,
          },
          'radius': (radius * 1.5).clamp(0, 50000),
        }
      }
    };

    if (parsedQuery.openNow) body['openNow'] = true;

    final res = await http.post(
      Uri.parse('https://places.googleapis.com/v1/places:searchText'),
      headers: {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': _apiKey,
        'X-Goog-FieldMask': fieldMask,
      },
      body: jsonEncode(body),
    );

    if (res.statusCode != 200) {
      throw Exception('Places API error: ${res.statusCode}');
    }

    final data = jsonDecode(res.body);
    final places = (data['places'] as List?)
            ?.map((p) => Place.fromJson(p as Map<String, dynamic>, center))
            .toList() ??
        [];

    // Filter by radius and min rating
    return places.where((p) {
      if (p.distance > radius) return false;
      if (parsedQuery.minRating != null &&
          (p.rating == null || p.rating! < parsedQuery.minRating!)) {
        return false;
      }
      return true;
    }).toList()
      ..sort((a, b) => a.distance.compareTo(b.distance));
  }
}
