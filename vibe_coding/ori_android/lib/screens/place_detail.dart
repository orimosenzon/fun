import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:share_plus/share_plus.dart';
import '../models/place.dart';
import '../providers/app_state.dart';
import '../l10n/translations.dart';

class PlaceDetailSheet extends StatelessWidget {
  final Place place;

  const PlaceDetailSheet({super.key, required this.place});

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final lang = state.lang;
    final t = (String key) => AppTranslations.tr(lang, key);
    final theme = Theme.of(context);
    final distKm = (place.distance / 1000).toStringAsFixed(1);

    return DraggableScrollableSheet(
      initialChildSize: 0.85,
      minChildSize: 0.5,
      maxChildSize: 0.95,
      builder: (context, scrollController) => Container(
        decoration: BoxDecoration(
          color: theme.scaffoldBackgroundColor,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(0),
          children: [
            // Drag handle
            Center(
              child: Container(
                margin: const EdgeInsets.only(top: 12, bottom: 8),
                width: 40,
                height: 5,
                decoration: BoxDecoration(
                  color: Colors.grey[400],
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
            ),

            // Header
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      place.name,
                      style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    icon: Icon(
                      state.isFavorite(place.name) ? Icons.star : Icons.star_border,
                      color: state.isFavorite(place.name) ? Colors.amber : null,
                    ),
                    onPressed: () => state.toggleFavorite(place),
                  ),
                ],
              ),
            ),

            // Photo
            if (place.photoUrl != null)
              Padding(
                padding: const EdgeInsets.all(16),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.network(
                    place.photoUrl!,
                    height: 200,
                    width: double.infinity,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => const SizedBox.shrink(),
                  ),
                ),
              ),

            // Info section
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Rating
                  if (place.rating != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          const Icon(Icons.star, color: Colors.amber, size: 20),
                          const SizedBox(width: 4),
                          Text(
                            '${place.rating!.toStringAsFixed(1)} (${place.userRatingCount ?? 0} ${t('reviews')})',
                            style: theme.textTheme.bodyMedium,
                          ),
                        ],
                      ),
                    ),

                  // Open status
                  if (place.isOpen != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          Icon(Icons.access_time,
                              size: 20,
                              color: place.isOpen! ? Colors.green : Colors.red),
                          const SizedBox(width: 4),
                          Text(
                            place.isOpen! ? t('openNow') : t('closed'),
                            style: TextStyle(
                              color: place.isOpen! ? Colors.green : Colors.red,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),

                  // Address
                  _infoRow(Icons.location_on, '${place.address} ($distKm ${t('km')})'),

                  // Phone
                  if (place.phoneNumber != null)
                    InkWell(
                      onTap: () => launchUrl(Uri.parse('tel:${place.phoneNumber}')),
                      child: _infoRow(Icons.phone, place.phoneNumber!, linkColor: theme.colorScheme.primary),
                    ),

                  // Website
                  if (place.websiteUri != null)
                    InkWell(
                      onTap: () => launchUrl(Uri.parse(place.websiteUri!), mode: LaunchMode.externalApplication),
                      child: _infoRow(Icons.language, t('website'), linkColor: theme.colorScheme.primary),
                    ),

                  const SizedBox(height: 16),

                  // Opening hours
                  if (place.weekdayDescriptions != null && place.weekdayDescriptions!.isNotEmpty) ...[
                    Text(t('openingHours'),
                        style: theme.textTheme.titleSmall?.copyWith(
                          color: theme.colorScheme.primary,
                          fontWeight: FontWeight.w600,
                        )),
                    const SizedBox(height: 8),
                    ...place.weekdayDescriptions!.asMap().entries.map((entry) {
                      final todayIdx = (DateTime.now().weekday % 7); // Sunday=0
                      final isoToGoogleDay = [6, 0, 1, 2, 3, 4, 5];
                      final isToday = entry.key == isoToGoogleDay[DateTime.now().weekday % 7];
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 4),
                        child: Text(
                          entry.value,
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: isToday ? FontWeight.bold : FontWeight.normal,
                          ),
                        ),
                      );
                    }),
                    const SizedBox(height: 16),
                  ],

                  // Action buttons
                  Text(t('navigation'),
                      style: theme.textTheme.titleSmall?.copyWith(
                        color: theme.colorScheme.primary,
                        fontWeight: FontWeight.w600,
                      )),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          icon: const Icon(Icons.directions_car, size: 18),
                          label: Text(t('directions')),
                          onPressed: () => _openMapsDirections(place),
                        ),
                      ),
                      const SizedBox(width: 8),
                      if (place.googleMapsUri != null)
                        Expanded(
                          child: OutlinedButton.icon(
                            icon: const Icon(Icons.map, size: 18),
                            label: const Text('Google Maps'),
                            onPressed: () => launchUrl(
                              Uri.parse(place.googleMapsUri!),
                              mode: LaunchMode.externalApplication,
                            ),
                          ),
                        ),
                    ],
                  ),

                  const SizedBox(height: 16),

                  // Share section
                  Text(t('shareSection'),
                      style: theme.textTheme.titleSmall?.copyWith(
                        color: theme.colorScheme.primary,
                        fontWeight: FontWeight.w600,
                      )),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          icon: const Icon(Icons.share, size: 18),
                          label: Text(t('share')),
                          onPressed: () => _sharePlace(place),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: OutlinedButton.icon(
                          icon: const Icon(Icons.content_copy, size: 18),
                          label: Text(t('copyLink')),
                          onPressed: () {
                            final url = place.googleMapsUri ??
                                'https://www.google.com/maps/search/?api=1&query=${place.location.latitude},${place.location.longitude}';
                            Clipboard.setData(ClipboardData(text: url));
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text(t('linkCopied')), duration: const Duration(seconds: 2)),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoRow(IconData icon, String text, {Color? linkColor}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: linkColor ?? Colors.grey),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: TextStyle(fontSize: 14, color: linkColor)),
          ),
        ],
      ),
    );
  }

  void _openMapsDirections(Place place) {
    final url =
        'https://www.google.com/maps/dir/?api=1&destination=${place.location.latitude},${place.location.longitude}';
    launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
  }

  void _sharePlace(Place place) {
    final url = place.googleMapsUri ??
        'https://www.google.com/maps/search/?api=1&query=${place.location.latitude},${place.location.longitude}';
    Share.share('${place.name}\n${place.address}\n$url');
  }
}
