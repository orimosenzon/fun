"""מערכת הקואורדינטות של העולם.

מישור משיק מקומי סביב מרכז המושבה, ביחידות של מטרים.
בגודו: x = מזרח, y = למעלה, z = דרום (כלומר צפון הוא z שלילי).
על פני 4 ק"מ השגיאה של הקירוב הזה מתחת למילימטר, ואת אותה נוסחה
(הפוכה) מחשב גם הקוד בגודו כדי להציג קו רוחב ואורך ולפתוח את סטריט ויו.
"""
import math

# מרכז המושבה, בערך צומת הרחובות הרצל ודרך הבנים
LAT0 = 32.4735
LON0 = 34.9765

_A = 6378137.0
_E2 = 6.69437999014e-3
_s = math.sin(math.radians(LAT0))
# רדיוס העקמומיות לאורך המרידיאן ולאורך האנך הראשי, בקו הרוחב של המרכז
M_PER_DEG_LAT = math.radians(1) * _A * (1 - _E2) / (1 - _E2 * _s * _s) ** 1.5
M_PER_DEG_LON = math.radians(1) * _A / math.sqrt(1 - _E2 * _s * _s) * math.cos(math.radians(LAT0))

BOUNDARY_REL = 1392849  # פרדס חנה - כרכור ב-OSM


def to_local(lat, lon):
    """(lat, lon) -> (east, north) במטרים."""
    return (lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT


def to_latlon(east, north):
    return LAT0 + north / M_PER_DEG_LAT, LON0 + east / M_PER_DEG_LON
