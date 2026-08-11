/* ============================================================
   Google Analytics 4 — פרדספדיה
   להדבקה בדף  מדיה ויקי:Common.js
   ------------------------------------------------------------
   מזהה המדידה של נכס "פרדספדיה" ב-GA4: G-9PGTWDGBKR (זרם Pardespedia, 8/8/2026).
   עורכים מחוברים אינם נמדדים בכוונה, כך שהנתונים מתארים
   קוראים ולא אותנו. להסרת ההחרגה: למחוק את בדיקת wgUserName.
   ============================================================ */

( function () {
	var MEASUREMENT_ID = 'G-9PGTWDGBKR';

	// אין מדידה לעורכים מחוברים: אורי וחוה גולשים בוויקי כל היום,
	// ובלי ההחרגה הם היו מטים את כל הסטטיסטיקה.
	if ( mw.config.get( 'wgUserName' ) !== null ) {
		return;
	}

	// טעינת gtag.js
	var script = document.createElement( 'script' );
	script.async = true;
	script.src = 'https://www.googletagmanager.com/gtag/js?id=' + MEASUREMENT_ID;
	document.head.appendChild( script );

	window.dataLayer = window.dataLayer || [];
	function gtag() {
		window.dataLayer.push( arguments );
	}
	window.gtag = gtag;

	gtag( 'js', new Date() );
	gtag( 'config', MEASUREMENT_ID, {
		// שם הערך במקום הכתובת המקודדת (%D7%90...) — קריא בדוחות
		page_title: mw.config.get( 'wgPageName' ).replace( /_/g, ' ' ),
		// לא לאסוף פרסום/רימרקטינג — ויקי קהילתי, לא חנות
		allow_google_signals: false,
		allow_ad_personalization_signals: false
	} );
}() );
