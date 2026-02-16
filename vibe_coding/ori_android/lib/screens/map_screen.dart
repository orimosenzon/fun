import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../providers/app_state.dart';
import '../l10n/translations.dart';
import '../models/place.dart';
import 'place_detail.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  GoogleMapController? _mapController;
  final _searchController = TextEditingController();
  final _searchFocusNode = FocusNode();
  bool _showHistory = false;

  @override
  void dispose() {
    _searchController.dispose();
    _searchFocusNode.dispose();
    super.dispose();
  }

  void _doSearch(AppState state) {
    final query = _searchController.text.trim();
    if (query.isEmpty) return;
    _searchFocusNode.unfocus();
    setState(() => _showHistory = false);
    state.search(query);
  }

  Set<Marker> _buildMarkers(AppState state) {
    final markers = <Marker>{};

    markers.add(Marker(
      markerId: const MarkerId('center'),
      position: state.center,
      draggable: true,
      icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueAzure),
      zIndex: 100,
      onDragEnd: (pos) => state.setCenter(pos),
    ));

    for (int i = 0; i < state.results.length; i++) {
      final place = state.results[i];
      markers.add(Marker(
        markerId: MarkerId('place_$i'),
        position: place.location,
        icon: BitmapDescriptor.defaultMarkerWithHue(BitmapDescriptor.hueRed),
        zIndex: 50,
        infoWindow: InfoWindow(title: place.name),
        onTap: () {
          state.selectPlace(i);
          _mapController?.animateCamera(CameraUpdate.newLatLng(place.location));
        },
      ));
    }

    return markers;
  }

  Set<Circle> _buildCircles(AppState state) {
    return {
      Circle(
        circleId: const CircleId('radius'),
        center: state.center,
        radius: state.radius,
        fillColor: Colors.blue.withValues(alpha: 0.08),
        strokeColor: Colors.blue.withValues(alpha: 0.4),
        strokeWidth: 2,
      ),
    };
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final lang = state.lang;
    final t = (String key) => AppTranslations.tr(lang, key);
    final isRtl = AppTranslations.isRtl(lang);

    return Directionality(
      textDirection: isRtl ? TextDirection.rtl : TextDirection.ltr,
      child: Scaffold(
        body: Stack(
          children: [
            // Map
            GoogleMap(
              initialCameraPosition: CameraPosition(target: state.center, zoom: 13),
              onMapCreated: (controller) => _mapController = controller,
              markers: _buildMarkers(state),
              circles: _buildCircles(state),
              mapType: state.mapType,
              myLocationEnabled: true,
              myLocationButtonEnabled: false,
              zoomControlsEnabled: false,
              mapToolbarEnabled: false,
            ),

            // Map type toggle
            Positioned(
              top: MediaQuery.of(context).padding.top + 70,
              right: 12,
              child: Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _mapTypeButton(t('map'), MapType.normal, state),
                    _mapTypeButton(t('satellite'), MapType.satellite, state),
                  ],
                ),
              ),
            ),

            // Search bar
            Positioned(
              top: MediaQuery.of(context).padding.top + 8,
              left: 12,
              right: 12,
              child: Column(
                children: [
                  Card(
                    elevation: 6,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                      child: Row(
                        children: [
                          // Language
                          PopupMenuButton<String>(
                            onSelected: (l) => state.setLang(l),
                            itemBuilder: (_) => [
                              PopupMenuItem(value: 'he', child: Text('🇮🇱 עברית${lang == 'he' ? ' ✓' : ''}')),
                              PopupMenuItem(value: 'en', child: Text('🇬🇧 English${lang == 'en' ? ' ✓' : ''}')),
                              PopupMenuItem(value: 'es', child: Text('🇪🇸 Español${lang == 'es' ? ' ✓' : ''}')),
                            ],
                            child: Padding(
                              padding: const EdgeInsets.all(8),
                              child: Text(lang.toUpperCase(),
                                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
                            ),
                          ),
                          // Dark mode
                          IconButton(
                            icon: Icon(state.isDark ? Icons.light_mode : Icons.dark_mode, size: 22),
                            onPressed: () => state.toggleDarkMode(),
                          ),
                          // Search input
                          Expanded(
                            child: TextField(
                              controller: _searchController,
                              focusNode: _searchFocusNode,
                              textDirection: isRtl ? TextDirection.rtl : TextDirection.ltr,
                              decoration: InputDecoration(
                                hintText: t('searchPlaceholder'),
                                hintStyle: const TextStyle(fontSize: 13),
                                border: InputBorder.none,
                                contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                              ),
                              style: const TextStyle(fontSize: 14),
                              onSubmitted: (_) => _doSearch(state),
                              onTap: () {
                                if (_searchController.text.isEmpty && state.searchHistory.isNotEmpty) {
                                  setState(() => _showHistory = true);
                                }
                              },
                              onChanged: (val) {
                                setState(() => _showHistory = val.isEmpty && state.searchHistory.isNotEmpty);
                              },
                            ),
                          ),
                          // Search button
                          IconButton(
                            icon: const Icon(Icons.search),
                            onPressed: state.isLoading ? null : () => _doSearch(state),
                            color: Theme.of(context).colorScheme.primary,
                          ),
                        ],
                      ),
                    ),
                  ),
                  // History dropdown
                  if (_showHistory)
                    Card(
                      elevation: 4,
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: state.searchHistory
                            .map((h) => ListTile(
                                  dense: true,
                                  title: Text(h, style: const TextStyle(fontSize: 13)),
                                  leading: const Icon(Icons.history, size: 18),
                                  trailing: IconButton(
                                    icon: const Icon(Icons.close, size: 16),
                                    onPressed: () {
                                      state.removeFromHistory(h);
                                      setState(() {});
                                    },
                                  ),
                                  onTap: () {
                                    _searchController.text = h;
                                    setState(() => _showHistory = false);
                                    _doSearch(state);
                                  },
                                ))
                            .toList(),
                      ),
                    ),
                ],
              ),
            ),

            // Loading
            if (state.isLoading) const Center(child: CircularProgressIndicator()),

            // Bottom sheet with results
            DraggableScrollableSheet(
              initialChildSize: 0.35,
              minChildSize: 0.08,
              maxChildSize: 0.85,
              snap: true,
              snapSizes: const [0.08, 0.35, 0.85],
              builder: (context, scrollController) => Container(
                decoration: BoxDecoration(
                  color: Theme.of(context).scaffoldBackgroundColor,
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
                  boxShadow: [
                    BoxShadow(color: Colors.black.withValues(alpha: 0.15), blurRadius: 12, offset: const Offset(0, -4)),
                  ],
                ),
                child: ListView(
                  controller: scrollController,
                  padding: EdgeInsets.zero,
                  children: [
                    // Handle
                    Center(
                      child: Container(
                        margin: const EdgeInsets.only(top: 10, bottom: 6),
                        width: 36,
                        height: 5,
                        decoration: BoxDecoration(color: Colors.grey[400], borderRadius: BorderRadius.circular(3)),
                      ),
                    ),
                    // Category chips
                    SizedBox(
                      height: 40,
                      child: ListView(
                        scrollDirection: Axis.horizontal,
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        children: (AppTranslations.categories[lang] ?? AppTranslations.categories['en']!)
                            .asMap()
                            .entries
                            .map((entry) {
                          final cat = entry.value;
                          final catKeys = AppTranslations.categoryKeys[lang]!;
                          final label = t(catKeys[entry.key]);
                          return Padding(
                            padding: const EdgeInsetsDirectional.only(end: 6),
                            child: ActionChip(
                              avatar: Text(cat['icon']!, style: const TextStyle(fontSize: 14)),
                              label: Text(label, style: const TextStyle(fontSize: 12)),
                              onPressed: () {
                                _searchController.text = cat['query']!;
                                _doSearch(state);
                              },
                            ),
                          );
                        }).toList(),
                      ),
                    ),
                    // Radius + sort
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      child: Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(t('radiusLabel'), style: const TextStyle(fontSize: 12, color: Colors.grey)),
                                    Text(
                                      state.radius >= 1000
                                          ? '${(state.radius / 1000).toStringAsFixed(state.radius % 1000 == 0 ? 0 : 1)} ${t('km')}'
                                          : '${state.radius.toInt()} ${t('m')}',
                                      style: TextStyle(
                                          fontSize: 12,
                                          fontWeight: FontWeight.w600,
                                          color: Theme.of(context).colorScheme.primary),
                                    ),
                                  ],
                                ),
                                Slider(value: state.radius, min: 200, max: 3800, divisions: 18, onChanged: (v) => state.setRadius(v)),
                              ],
                            ),
                          ),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(t('sortLabel'), style: const TextStyle(fontSize: 12, color: Colors.grey)),
                              const SizedBox(height: 4),
                              DropdownButton<String>(
                                value: state.sortBy,
                                isDense: true,
                                style: TextStyle(fontSize: 13, color: Theme.of(context).textTheme.bodyMedium?.color),
                                items: [
                                  DropdownMenuItem(value: 'distance', child: Text(t('sortDistance'))),
                                  DropdownMenuItem(value: 'rating', child: Text(t('sortRating'))),
                                  DropdownMenuItem(value: 'reviews', child: Text(t('sortReviews'))),
                                ],
                                onChanged: (v) => state.setSortBy(v!),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const Divider(height: 1),
                    // Status or results
                    if (state.results.isEmpty && !state.isLoading)
                      Padding(
                        padding: const EdgeInsets.all(24),
                        child: Text(
                          state.errorMessage != null ? '${t('searchError')}${state.errorMessage}' : t('statusDefault'),
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Colors.grey[600], fontSize: 14),
                        ),
                      ),
                    ...state.results.asMap().entries.map((entry) {
                      final i = entry.key;
                      final place = entry.value;
                      return _ResultCard(
                        place: place,
                        index: i,
                        isSelected: state.selectedIndex == i,
                        onTap: () {
                          state.selectPlace(i);
                          _mapController?.animateCamera(CameraUpdate.newLatLngZoom(place.location, 16));
                        },
                        onDetailTap: () => _showDetail(context, place),
                        onNavTap: () => _openNavigation(place),
                      );
                    }),
                    const SizedBox(height: 80),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _mapTypeButton(String label, MapType type, AppState state) {
    final isActive = state.mapType == type;
    return InkWell(
      onTap: () => state.setMapType(type),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: isActive ? Theme.of(context).colorScheme.primary : Colors.transparent,
          borderRadius: BorderRadius.circular(6),
        ),
        child: Text(label,
            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: isActive ? Colors.white : null)),
      ),
    );
  }

  void _showDetail(BuildContext context, Place place) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ChangeNotifierProvider.value(
        value: context.read<AppState>(),
        child: PlaceDetailSheet(place: place),
      ),
    );
  }

  void _openNavigation(Place place) {
    final url =
        'https://www.google.com/maps/dir/?api=1&destination=${place.location.latitude},${place.location.longitude}';
    launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
  }
}

class _ResultCard extends StatelessWidget {
  final Place place;
  final int index;
  final bool isSelected;
  final VoidCallback onTap;
  final VoidCallback onDetailTap;
  final VoidCallback onNavTap;

  const _ResultCard({
    required this.place,
    required this.index,
    required this.isSelected,
    required this.onTap,
    required this.onDetailTap,
    required this.onNavTap,
  });

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final t = (String key) => AppTranslations.tr(state.lang, key);
    final theme = Theme.of(context);
    final distKm = (place.distance / 1000).toStringAsFixed(1);

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      elevation: isSelected ? 4 : 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: isSelected ? BorderSide(color: theme.colorScheme.primary, width: 2) : BorderSide.none,
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Photo
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: place.photoUrl != null
                    ? Image.network(place.photoUrl!, width: 68, height: 68, fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => _placeholderPhoto())
                    : _placeholderPhoto(),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(place.name,
                              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis),
                        ),
                        InkWell(
                          onTap: () => state.toggleFavorite(place),
                          child: Icon(
                            state.isFavorite(place.name) ? Icons.star : Icons.star_border,
                            size: 20,
                            color: state.isFavorite(place.name) ? Colors.amber : Colors.grey,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(place.address,
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        if (place.rating != null) ...[
                          const Icon(Icons.star, size: 14, color: Colors.amber),
                          const SizedBox(width: 2),
                          Text('${place.rating!.toStringAsFixed(1)}',
                              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500)),
                          Text(' (${place.userRatingCount ?? 0})',
                              style: TextStyle(fontSize: 11, color: Colors.grey[500])),
                          const SizedBox(width: 8),
                        ],
                        if (place.isOpen != null) ...[
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                            decoration: BoxDecoration(
                              color: (place.isOpen! ? Colors.green : Colors.red).withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(place.isOpen! ? t('open') : t('closed'),
                                style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w500,
                                    color: place.isOpen! ? Colors.green : Colors.red)),
                          ),
                          const SizedBox(width: 8),
                        ],
                        Text('$distKm ${t('km')}', style: TextStyle(fontSize: 11, color: Colors.grey[500])),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        _actionButton(icon: Icons.directions_car, label: t('navigate'), filled: true, onTap: onNavTap, context: context),
                        const SizedBox(width: 6),
                        _actionButton(icon: Icons.info_outline, label: t('details'), onTap: onDetailTap, context: context),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _placeholderPhoto() {
    return Container(
      width: 68,
      height: 68,
      decoration: BoxDecoration(color: Colors.grey[200], borderRadius: BorderRadius.circular(8)),
      child: const Icon(Icons.place, color: Colors.grey, size: 28),
    );
  }

  Widget _actionButton({
    required IconData icon,
    required String label,
    bool filled = false,
    required VoidCallback onTap,
    required BuildContext context,
  }) {
    final primary = Theme.of(context).colorScheme.primary;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(6),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: filled ? primary : Colors.transparent,
          border: Border.all(color: filled ? primary : Colors.grey[300]!),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: filled ? Colors.white : Colors.grey[600]),
            const SizedBox(width: 4),
            Text(label,
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: filled ? Colors.white : Colors.grey[600])),
          ],
        ),
      ),
    );
  }
}
