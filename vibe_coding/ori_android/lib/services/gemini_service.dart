import 'dart:convert';
import 'package:http/http.dart' as http;

class ParsedQuery {
  final String textQuery;
  final bool openNow;
  final double? minRating;
  final List<String> keywords;

  ParsedQuery({
    required this.textQuery,
    this.openNow = false,
    this.minRating,
    this.keywords = const [],
  });
}

class GeminiService {
  static const _apiKey = 'AIzaSyDJOFb9xXjwtHDLMrBIeBRPuSQpWnjTUo4';

  static const Map<String, String> _prompts = {
    'en': '''You are a business search query analyzer in English. Parse the following query and extract structured information.

Query: "__QUERY__"

Return JSON only in the following format (no additional text):
{
  "textQuery": "the main search term - the primary type of business",
  "openNow": true/false (is the user looking for places open now),
  "minRating": null or number (if a minimum rating was specified),
  "keywords": ["additional keywords"]
}''',
    'he': '''אתה מנתח שאילתות חיפוש עסקים בעברית. קבל את השאילתה הבאה וחלץ ממנה מידע מובנה.

שאילתה: "__QUERY__"

החזר JSON בלבד בפורמט הבא (ללא טקסט נוסף):
{
  "textQuery": "מונח החיפוש המרכזי - הסוג העיקרי של העסק",
  "openNow": true/false (האם המשתמש מחפש מקומות פתוחים עכשיו),
  "minRating": null או מספר (אם צוין דירוג מינימלי),
  "keywords": ["מילות מפתח נוספות"]
}''',
    'es': '''Eres un analizador de consultas de búsqueda de negocios en español. Analiza la siguiente consulta y extrae información estructurada.

Consulta: "__QUERY__"

Devuelve solo JSON en el siguiente formato (sin texto adicional):
{
  "textQuery": "el término de búsqueda principal - el tipo principal de negocio",
  "openNow": true/false (si el usuario busca lugares abiertos ahora),
  "minRating": null o número (si se especificó una calificación mínima),
  "keywords": ["palabras clave adicionales"]
}''',
  };

  static Future<ParsedQuery> parseQuery(String query, String lang) async {
    final prompt = (_prompts[lang] ?? _prompts['en']!).replaceAll('__QUERY__', query);

    try {
      final res = await http.post(
        Uri.parse(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$_apiKey'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'contents': [
            {
              'parts': [
                {'text': prompt}
              ]
            }
          ]
        }),
      );

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final text = data['candidates'][0]['content']['parts'][0]['text'] as String;
        final jsonMatch = RegExp(r'\{[\s\S]*\}').firstMatch(text);
        if (jsonMatch != null) {
          final parsed = jsonDecode(jsonMatch.group(0)!) as Map<String, dynamic>;
          return ParsedQuery(
            textQuery: parsed['textQuery'] ?? query,
            openNow: parsed['openNow'] ?? false,
            minRating: (parsed['minRating'] as num?)?.toDouble(),
            keywords: (parsed['keywords'] as List?)?.map((e) => e.toString()).toList() ?? [],
          );
        }
      }
    } catch (e) {
      // Fall back to raw query
    }
    return ParsedQuery(textQuery: query);
  }
}
