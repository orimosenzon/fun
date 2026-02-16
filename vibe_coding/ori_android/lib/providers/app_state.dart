import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/place.dart';
import '../services/gemini_service.dart';
import '../services/places_service.dart';
import '../services/location_service.dart';

class AppState extends ChangeNotifier {
  // Language
  String _lang = 'en';
  String get lang => _lang;

  // Theme
  bool _isDark = false;
  bool get isDark => _isDark;

  // Location & search
  LatLng _center = LocationService.defaultLocation;
  LatLng get center => _center;

  LatLng? _userLocation;
  LatLng? get userLocation => _userLocation;

  double _radius = 2000;
  double get radius => _radius;

  // Results
  List<Place> _results = [];
  List<Place> get results => _results;

  bool _isLoading = false;
  bool get isLoading => _isLoading;

  String? _errorMessage;
  String? get errorMessage => _errorMessage;

  // Sort
  String _sortBy = 'distance';
  String get sortBy => _sortBy;

  // Favorites
  List<Map<String, dynamic>> _favorites = [];
  List<Map<String, dynamic>> get favorites => _favorites;

  // Search history
  List<String> _searchHistory = [];
  List<String> get searchHistory => _searchHistory;

  // Selected place index
  int? _selectedIndex;
  int? get selectedIndex => _selectedIndex;

  // Map type
  MapType _mapType = MapType.normal;
  MapType get mapType => _mapType;

  SharedPreferences? _prefs;

  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    _lang = _prefs?.getString('biz_lang') ?? 'en';
    _isDark = _prefs?.getBool('biz_dark') ?? false;
    _favorites = (_prefs?.getStringList('biz_favorites') ?? [])
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .toList();
    _searchHistory = _prefs?.getStringList('biz_history') ?? [];

    // Get user location
    _center = await LocationService.getCurrentLocation();
    _userLocation = _center;
    notifyListeners();
  }

  void setLang(String lang) {
    _lang = lang;
    _prefs?.setString('biz_lang', lang);
    notifyListeners();
  }

  void toggleDarkMode() {
    _isDark = !_isDark;
    _prefs?.setBool('biz_dark', _isDark);
    notifyListeners();
  }

  void setCenter(LatLng center) {
    _center = center;
    notifyListeners();
  }

  void setRadius(double radius) {
    _radius = radius;
    notifyListeners();
  }

  void setSortBy(String sortBy) {
    _sortBy = sortBy;
    _sortResults();
    notifyListeners();
  }

  void setMapType(MapType type) {
    _mapType = type;
    notifyListeners();
  }

  void selectPlace(int? index) {
    _selectedIndex = index;
    notifyListeners();
  }

  // Favorites
  bool isFavorite(String name) {
    return _favorites.any((f) => f['id'] == name);
  }

  void toggleFavorite(Place place) {
    final idx = _favorites.indexWhere((f) => f['id'] == place.name);
    if (idx >= 0) {
      _favorites.removeAt(idx);
    } else {
      _favorites.add({
        'id': place.name,
        'name': place.name,
        'address': place.address,
        'lat': place.location.latitude,
        'lng': place.location.longitude,
      });
    }
    _prefs?.setStringList(
      'biz_favorites',
      _favorites.map((f) => jsonEncode(f)).toList(),
    );
    notifyListeners();
  }

  // Search history
  void _addToHistory(String query) {
    _searchHistory.remove(query);
    _searchHistory.insert(0, query);
    if (_searchHistory.length > 10) _searchHistory.removeLast();
    _prefs?.setStringList('biz_history', _searchHistory);
  }

  void removeFromHistory(String query) {
    _searchHistory.remove(query);
    _prefs?.setStringList('biz_history', _searchHistory);
    notifyListeners();
  }

  void _sortResults() {
    switch (_sortBy) {
      case 'rating':
        _results.sort((a, b) => (b.rating ?? 0).compareTo(a.rating ?? 0));
        break;
      case 'reviews':
        _results.sort((a, b) => (b.userRatingCount ?? 0).compareTo(a.userRatingCount ?? 0));
        break;
      default:
        _results.sort((a, b) => a.distance.compareTo(b.distance));
    }
  }

  // Main search
  Future<void> search(String query) async {
    if (query.trim().isEmpty) return;

    _isLoading = true;
    _errorMessage = null;
    _selectedIndex = null;
    _results = [];
    notifyListeners();

    _addToHistory(query.trim());

    try {
      final parsed = await GeminiService.parseQuery(query.trim(), _lang);
      final places = await PlacesService.searchPlaces(
        parsedQuery: parsed,
        center: _center,
        radius: _radius,
        lang: _lang,
      );
      _results = places;
      _sortResults();
    } catch (e) {
      _errorMessage = e.toString();
    }

    _isLoading = false;
    notifyListeners();
  }
}
