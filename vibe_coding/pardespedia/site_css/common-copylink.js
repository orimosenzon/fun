/* ============================================================
   כפתור "העתק קישור" — קישור קצר וקריא לערך
   ------------------------------------------------------------
   הבעיה: כשמעתיקים כתובת של ערך משורת הכתובת, כרום מקודד את
   העברית ל-%D7%A4%D7%95... והקישור מתנפח מ-56 תווים ל-166.
   הכתובת עצמה בסדר גמור — רק ההעתקה מכערת אותה.

   הפתרון: כפתור שמעתיק את הכתובת בעברית, לא מקודדת. היא עובדת
   בכל דפדפן, נראית טוב בוואטסאפ, ואומרת לקורא מה הוא עומד לפתוח.
   (הקצרה עוד יותר היא ?curid=<מזהה>, 36 תווים, אבל היא אטומה
   ולא מרמזת על התוכן. אם תרצו אותה, זו שורה אחת כאן.)

   מופיע רק בערכי תוכן (מרחב ראשי, תצוגה רגילה).
   ============================================================ */

( function () {
	var cfg = mw.config.get( [ 'wgArticleId', 'wgNamespaceNumber', 'wgPageName', 'wgAction' ] );

	// לא בדפי מיוחד, קטגוריה, שיחה וכו', לא בעריכה, ולא בדף שאינו קיים
	if ( cfg.wgArticleId <= 0 || cfg.wgNamespaceNumber !== 0 || cfg.wgAction !== 'view' ) {
		return;
	}

	// wgPageName כבר מגיע עם קווים תחתונים ובעברית לא מקודדת — בדיוק מה שרצינו
	var url = location.origin + '/wiki/' + cfg.wgPageName;

	var btn = document.createElement( 'button' );
	btn.type = 'button';
	btn.id = 'pp-copy-link';
	btn.title = 'העתקת כתובת הערך, בעברית ובלי הקידוד הארוך';
	btn.textContent = '🔗 העתק קישור';

	function flash( text ) {
		var was = btn.textContent;
		btn.textContent = text;
		btn.disabled = true;
		setTimeout( function () {
			btn.textContent = was;
			btn.disabled = false;
		}, 1800 );
	}

	// navigator.clipboard קיים רק ב-HTTPS ובדפדפן עדכני. הנפילה לאחור
	// היא textarea זמני, שעובד גם בדפדפנים ישנים ובחיבור לא מאובטח.
	function copy() {
		if ( navigator.clipboard && navigator.clipboard.writeText ) {
			navigator.clipboard.writeText( url ).then(
				function () { flash( '✓ הועתק' ); },
				fallback
			);
		} else {
			fallback();
		}
	}

	function fallback() {
		var ta = document.createElement( 'textarea' );
		ta.value = url;
		ta.setAttribute( 'readonly', '' );
		ta.style.position = 'fixed';
		ta.style.top = '-1000px';
		document.body.appendChild( ta );
		ta.select();
		var ok = false;
		try {
			ok = document.execCommand( 'copy' );
		} catch ( e ) {
			ok = false;
		}
		document.body.removeChild( ta );
		// אם גם זה נכשל, עדיף להראות את הכתובת מאשר להיכשל בשקט:
		// הקורא יכול לסמן ולהעתיק בעצמו.
		flash( ok ? '✓ הועתק' : url );
	}

	btn.addEventListener( 'click', copy );

	// עוגן: כותרת הערך. עורות שונים בונים את הסביבה שלה אחרת,
	// ולכן נצמדים לכותרת עצמה ולא לתפריט צד כלשהו.
	function place() {
		var h = document.getElementById( 'firstHeading' );
		if ( !h || document.getElementById( 'pp-copy-link' ) ) {
			return;
		}
		var wrap = document.createElement( 'div' );
		wrap.className = 'pp-copy-link-wrap';
		wrap.appendChild( btn );
		h.parentNode.insertBefore( wrap, h.nextSibling );
	}

	if ( document.readyState === 'loading' ) {
		document.addEventListener( 'DOMContentLoaded', place );
	} else {
		place();
	}
}() );
