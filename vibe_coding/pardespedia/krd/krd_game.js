/* ============================================================
   משחק "למי מגיע הקרדיט?" — תבנית:למי מגיע הקרדיט
   ------------------------------------------------------------
   התוכן כולו כתוב בוויקיטקסט, בתבניות (שאלות, אירועי ציר הזמן,
   תקופות ראשי המועצה). הקוד כאן רק הופך אותו למשחק: שאלה אחרי
   שאלה, ציון ופירוט בסוף, ואז ציר זמן גרפי והסיפור המלא.
   בלי הקוד הזה הדף עדיין קריא: השאלות עם תשובה מתקפלת והציר
   כרשימה.
   ============================================================ */

( function () {
	var PALETTE = [ '#b5651d', '#4f7cac', '#6a994e', '#8e5572', '#2a9d8f',
		'#c08a1e', '#5e548e', '#c1666b', '#3d5a80', '#7f6a3a', '#9a5b8f', '#4d8b6f' ];
	var BROWN = '#7a5c3e';

	var CSS = [
		'.krd-js .krd-btn{cursor:pointer;border:0;border-radius:10px;padding:10px 18px;font-size:105%;font-weight:bold;background:' + BROWN + ';color:#fff;font-family:inherit}',
		'.krd-js .krd-btn:hover{filter:brightness(1.12)}',
		'.krd-js .krd-btn-light{background:#fff;color:' + BROWN + ';border:1px solid #d9c9b0}',
		'.krd-js .krd-choice{display:block;width:100%;text-align:right;cursor:pointer;background:#fff;border:1px solid #d9c9b0;border-radius:10px;padding:10px 14px;margin:7px 0;font-size:102%;font-family:inherit;color:#222;line-height:1.45}',
		'.krd-js .krd-choice:hover:not([disabled]){border-color:' + BROWN + ';background:#fffaf2}',
		'.krd-js .krd-choice[disabled]{cursor:default}',
		'.krd-js .krd-choice.krd-right{background:#e7f5ea;border-color:#34a853}',
		'.krd-js .krd-choice.krd-wrong{background:#fdecea;border-color:#d93025}',
		'.krd-js .krd-choice.krd-dim{opacity:.55}',
		'.krd-js .krd-fade{animation:krdIn .25s ease}',
		'@keyframes krdIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}'
	].join( '\n' );

	function el( tag, style, html ) {
		var e = document.createElement( tag );
		if ( style ) { e.style.cssText = style; }
		if ( html !== undefined && html !== null ) { e.innerHTML = html; }
		return e;
	}
	function txt( n ) {
		return ( ( n && n.textContent ) || '' ).replace( /\s+/g, ' ' ).trim();
	}
	// HTML של אלמנט, בלי כפתור הקיפול שמדיה ויקי הוסיף לו (אם הוסיף)
	function html( n ) {
		if ( !n ) { return ''; }
		var c = n.cloneNode( true );
		$( c ).find( '.mw-collapsible-toggle' ).remove();
		var inner = $( c ).find( '.mw-collapsible-content' );
		return inner.length ? inner.html() : c.innerHTML;
	}
	function firstYear( s ) {
		var m = ( s || '' ).match( /\d{4}(\.\d+)?/ );
		return m ? parseFloat( m[ 0 ] ) : NaN;
	}
	function esc( s ) {
		return String( s ).replace( /[&<>"]/g, function ( c ) {
			return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ c ];
		} );
	}

	/* ---------- קריאת הנתונים מהדף ---------- */

	function readPeriods( root ) {
		var colors = {}, n = 0;
		return $( root ).find( '.krd-period' ).map( function () {
			var name = txt( this );
			if ( !( name in colors ) ) { colors[ name ] = PALETTE[ n++ % PALETTE.length ]; }
			return {
				from: parseFloat( this.getAttribute( 'data-from' ) ),
				// "עד" ריק: עד היום, והפס נמשך עד קצה הציר
				to: parseFloat( this.getAttribute( 'data-to' ) ) || Infinity,
				name: name,
				color: colors[ name ]
			};
		} ).get();
	}
	function periodAt( periods, y, name ) {
		var i;
		if ( name ) {
			for ( i = 0; i < periods.length; i++ ) { if ( periods[ i ].name === name ) { return periods[ i ]; } }
		}
		for ( i = 0; i < periods.length; i++ ) {
			if ( y >= periods[ i ].from && y < periods[ i ].to ) { return periods[ i ]; }
		}
		return null;
	}
	function readQuestions( root ) {
		return $( root ).find( '.krd-q' ).map( function () {
			var q = this, opts = [];
			$( q ).find( '.krd-opt' ).each( function () {
				var t = $( this ).find( '.krd-opt-text' )[ 0 ];
				if ( txt( t ) ) {
					opts.push( { key: this.getAttribute( 'data-key' ), html: t.innerHTML, text: txt( t ) } );
				}
			} );
			var yearLabel = txt( $( q ).find( '.krd-q-year' )[ 0 ] );
			return {
				yearLabel: yearLabel,
				year: firstYear( q.getAttribute( 'data-year' ) || yearLabel ),
				html: $( q ).find( '.krd-q-text' ).html(),
				opts: opts,
				correct: ( q.getAttribute( 'data-correct' ) || '' ).trim(),
				explain: html( $( q ).find( '.krd-explain-body' )[ 0 ] )
			};
		} ).get().filter( function ( q ) { return q.opts.length > 1; } );
	}
	function readEvents( root, periods ) {
		return $( root ).find( '.krd-ev' ).map( function () {
			var y = firstYear( this.getAttribute( 'data-year' ) );
			var head = ( this.getAttribute( 'data-head' ) || '' ).trim();
			return {
				node: this,
				year: y,
				date: txt( $( this ).find( '.krd-ev-date' )[ 0 ] ),
				title: txt( $( this ).find( '.krd-ev-title' )[ 0 ] ),
				// "ראש מועצה = -": אירוע שאין לייחס לאף ראש מועצה (למשל עשורים של הזנחה)
				period: head === '-' ? null : periodAt( periods, y, head )
			};
		} ).get().filter( function ( e ) { return !isNaN( e.year ); } );
	}

	/* ---------- גרפיקה: הציר ---------- */

	// מיקום על הציר, מימין לשמאל (השנה הראשונה בצד ימין, כמו קריאה בעברית)
	function pos( y, min, max ) {
		return Math.max( 0, Math.min( 100, ( y - min ) / ( max - min ) * 100 ) );
	}
	function ticks( min, max ) {
		var box = el( 'div', 'position:relative;height:18px;margin-top:3px;font-size:75%;color:#8a6a5c' );
		var step = ( max - min ) > 60 ? 10 : 5;
		for ( var y = Math.ceil( min / step ) * step; y <= max; y += step ) {
			var t = el( 'div', 'position:absolute;top:0;transform:translateX(50%);right:' + pos( y, min, max ) + '%' , String( y ) );
			box.appendChild( t );
		}
		return box;
	}

	// ציר ההתקדמות בזמן המשחק: שנים ונקודה לכל שאלה, בלי שמות (שלא יסגיר תשובות)
	function progressAxis( qs, cur, answers, min, max ) {
		var wrap = el( 'div', 'margin:6px 0 14px;padding:0 10px' );
		var line = el( 'div', 'position:relative;height:22px' );
		line.appendChild( el( 'div', 'position:absolute;left:0;right:0;top:10px;height:3px;border-radius:2px;background:#e3d5c0' ) );
		qs.forEach( function ( q, i ) {
			if ( isNaN( q.year ) ) { return; }
			var bg = i === cur ? BROWN : ( answers[ i ] === undefined ? '#fff' :
				( answers[ i ] === q.correct ? '#34a853' : '#d93025' ) );
			var size = i === cur ? 18 : 12;
			line.appendChild( el( 'div', 'position:absolute;transform:translateX(50%);right:' + pos( q.year, min, max ) +
				'%;top:' + ( 11.5 - size / 2 ) + 'px;width:' + size + 'px;height:' + size + 'px;border-radius:50%;background:' + bg +
				';border:2px solid ' + ( i === cur ? '#fbf6ee' : BROWN ) + ';box-sizing:border-box' + ( i === cur ? ';box-shadow:0 0 0 2px ' + BROWN : '' ) ) );
		} );
		wrap.appendChild( line );
		wrap.appendChild( ticks( min, max ) );
		return wrap;
	}

	// הציר המלא בסוף: פס תקופות של ראשי המועצה, ומעליו האירועים ממוספרים
	function eraTimeline( periods, events, min, max, widthPx ) {
		var wrap = el( 'div', 'margin:10px 0 18px;padding:0 12px' );

		// סימני האירועים, בשורות כדי שאירועים קרובים לא יעלו זה על זה
		var rows = [], placed = [];
		events.forEach( function ( e, i ) {
			var p = pos( e.year, min, max ), r = 0;
			// מרחק מינימלי בין סימנים באותה שורה: 25 פיקסלים, באחוזים של רוחב הציר בפועל
			var gap = 2500 / Math.max( 200, widthPx || 700 );
			while ( rows[ r ] !== undefined && Math.abs( rows[ r ] - p ) < gap ) { r++; }
			rows[ r ] = p;
			placed.push( { e: e, i: i, p: p, r: r } );
		} );
		var nRows = Math.max( 1, rows.length ), ROW = 26;
		var marks = el( 'div', 'position:relative;height:' + ( nRows * ROW + 6 ) + 'px' );
		placed.forEach( function ( m ) {
			var c = m.e.period ? m.e.period.color : BROWN;
			var top = ( nRows - 1 - m.r ) * ROW;
			marks.appendChild( el( 'div', 'position:absolute;width:2px;background:' + c + ';opacity:.5;transform:translateX(50%);right:' + m.p +
				'%;top:' + ( top + 20 ) + 'px;bottom:0' ) );
			var dot = el( 'div', 'position:absolute;transform:translateX(50%);right:' + m.p + '%;top:' + top +
				'px;width:21px;height:21px;line-height:21px;text-align:center;border-radius:50%;background:' + c +
				';color:#fff;font-size:72%;font-weight:bold;cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.25)', String( m.i + 1 ) );
			dot.title = m.e.date + ': ' + m.e.title;
			dot.onclick = function () { m.e.node.scrollIntoView( { behavior: 'smooth', block: 'center' } ); flash( m.e.node ); };
			marks.appendChild( dot );
		} );
		wrap.appendChild( marks );

		// פס התקופות
		var bar = el( 'div', 'position:relative;height:34px;border-radius:8px;overflow:hidden;background:#eee6d8' );
		periods.forEach( function ( p ) {
			var a = Math.max( p.from, min ), b = Math.min( p.to, max );
			if ( b <= a ) { return; }
			var w = pos( b, min, max ) - pos( a, min, max );
			var seg = el( 'div', 'position:absolute;top:0;bottom:0;right:' + pos( a, min, max ) + '%;width:' + w +
				'%;background:' + p.color + ';color:#fff;font-size:72%;font-weight:bold;display:flex;align-items:center;justify-content:center;' +
				'white-space:nowrap;overflow:hidden;border-left:1px solid rgba(255,255,255,.6)', w >= 10 ? esc( p.name ) : '' );
			seg.title = p.name + ' (' + Math.floor( p.from ) + '–' + ( p.to === Infinity ? 'היום' : Math.floor( p.to ) ) + ')';
			bar.appendChild( seg );
		} );
		wrap.appendChild( bar );
		wrap.appendChild( ticks( min, max ) );

		// מקרא: ראשי המועצה שבטווח, כולל הצרים שאין בהם מקום לשם
		var seen = {}, legend = el( 'div', 'display:flex;flex-wrap:wrap;gap:6px 12px;margin-top:8px;font-size:82%;color:#444' );
		periods.forEach( function ( p ) {
			if ( p.to <= min || p.from >= max || seen[ p.name ] ) { return; }
			seen[ p.name ] = 1;
			legend.appendChild( el( 'span', '', '<span style="display:inline-block;width:10px;height:10px;border-radius:3px;margin-left:4px;background:' +
				p.color + '"></span>' + esc( p.name ) ) );
		} );
		wrap.appendChild( legend );
		return wrap;
	}

	function flash( node ) {
		var old = node.style.background;
		node.style.transition = 'background .4s';
		node.style.background = '#fff3c4';
		setTimeout( function () { node.style.background = old; }, 1200 );
	}

	// מספר וצבע לכל כרטיס אירוע ברשימה, תואמים לסימנים בציר
	function decorateEvents( events ) {
		events.forEach( function ( e, i ) {
			var c = e.period ? e.period.color : BROWN;
			var dot = $( e.node ).find( '.krd-ev-dot' )[ 0 ];
			if ( dot ) {
				dot.style.background = c;
				dot.style.width = dot.style.height = '21px';
				dot.style.right = '-12px';
				dot.style.lineHeight = '21px';
				dot.style.textAlign = 'center';
				dot.style.color = '#fff';
				dot.style.fontSize = '72%';
				dot.style.fontWeight = 'bold';
				dot.textContent = String( i + 1 );
			}
			if ( e.period && !$( e.node ).find( '.krd-ev-chip' ).length ) {
				var chip = el( 'span', 'display:inline-block;margin-right:8px;padding:0 9px;border-radius:999px;font-size:78%;color:#fff;vertical-align:1px;background:' + c, esc( e.period.name ) );
				chip.className = 'krd-ev-chip';
				$( e.node ).find( '.krd-ev-date' ).append( chip );
			}
		} );
	}

	/* ---------- המשחק ---------- */

	function verdict( score, n ) {
		var r = score / n;
		if ( r === 1 ) { return '🏆 מושלם! את הסיפור הזה אפשר כבר לספר לאחרים.'; }
		if ( r >= 0.7 ) { return '👏 יפה מאוד! רוב הסיפור כבר מוכר לך.'; }
		if ( r >= 0.4 ) { return '🙂 לא רע, ונשאר עוד מה לגלות.'; }
		return '📚 הסיפור כנראה מפתיע. הוא מחכה למטה.';
	}

	function init( root ) {
		if ( root.getAttribute( 'data-krd' ) ) { return; }
		root.setAttribute( 'data-krd', '1' );

		var periods = readPeriods( root );
		var qs = readQuestions( root );
		var events = readEvents( root, periods );
		if ( !qs.length ) { return; }

		var years = qs.map( function ( q ) { return q.year; } )
			.concat( events.map( function ( e ) { return e.year; } ) )
			.filter( function ( y ) { return !isNaN( y ); } );
		var first = Math.floor( Math.min.apply( null, years ) ), last = Math.ceil( Math.max.apply( null, years ) );
		var min = first - 2, max = last + 1;

		var $static = $( root ).find( '.krd-questions, .krd-nojs, .krd-timeline, .krd-story' );
		var $timeline = $( root ).find( '.krd-timeline' ), $story = $( root ).find( '.krd-story' );
		$( root ).addClass( 'krd-js' );
		$static.hide();

		var stage = el( 'div', 'margin-top:10px' );
		$( root ).find( '.krd-head' ).after( stage );

		var answers, cur;

		// גלילה לראש המשחק, מתחת לסרגל העליון הדביק של האתר
		function top() {
			// הסרגל הוא מה שנמצא בראש החלון ומיקומו fixed או sticky
			var off = 10, n = document.elementFromPoint( window.innerWidth / 2, 3 );
			while ( n && n !== document.body ) {
				var ps = getComputedStyle( n ).position;
				if ( ps === 'fixed' || ps === 'sticky' ) { off += n.getBoundingClientRect().bottom; break; }
				n = n.parentElement;
			}
			window.scrollTo( { top: root.getBoundingClientRect().top + window.pageYOffset - off, behavior: 'smooth' } );
		}

		function start() {
			answers = [];
			cur = 0;
			$timeline.hide();
			$story.hide();
			$( root ).find( '.krd-era' ).remove();
			question();
		}

		function intro() {
			stage.innerHTML = '';
			var box = el( 'div', 'text-align:center;padding:10px 0 4px' );
			box.className = 'krd-fade';
			box.appendChild( progressAxis( qs, -1, [], min, max ) );
			box.appendChild( el( 'div', 'margin:6px 0 14px;color:#5a4632;line-height:1.5',
				qs.length + ' שאלות לאורך ' + ( last - first ) + ' שנים, מ-' + first + ' ועד היום. ' +
				'בסוף: ציון, התשובות הנכונות, ציר הזמן המלא והסיפור.' ) );
			var b = el( 'button', '', 'מתחילים ›' );
			b.className = 'krd-btn';
			b.onclick = start;
			box.appendChild( b );
			stage.appendChild( box );
		}

		function question() {
			var q = qs[ cur ];
			stage.innerHTML = '';
			var box = el( 'div' );
			box.className = 'krd-fade';
			box.appendChild( el( 'div', 'display:flex;justify-content:space-between;align-items:center;font-size:85%;color:#8a6a5c',
				'<span>שאלה ' + ( cur + 1 ) + ' מתוך ' + qs.length + '</span><span>' +
				'ציון: ' + answers.filter( function ( a, i ) { return a === qs[ i ].correct; } ).length + '</span>' ) );
			box.appendChild( progressAxis( qs, cur, answers, min, max ) );
			if ( q.yearLabel ) {
				box.appendChild( el( 'span', 'display:inline-block;background:' + BROWN + ';color:#fff;border-radius:999px;padding:1px 12px;font-size:88%;font-weight:bold', esc( q.yearLabel ) ) );
			}
			box.appendChild( el( 'div', 'font-size:112%;font-weight:bold;margin:8px 0 6px;line-height:1.5', q.html ) );
			var buttons = [];
			q.opts.forEach( function ( o ) {
				var b = el( 'button', '', '<b>' + esc( o.key ) + '.</b> ' + o.html );
				b.className = 'krd-choice';
				b.onclick = function () { answer( o.key, buttons, box ); };
				b.setAttribute( 'data-key', o.key );
				buttons.push( b );
				box.appendChild( b );
			} );
			stage.appendChild( box );
		}

		function answer( key, buttons, box ) {
			var q = qs[ cur ];
			answers[ cur ] = key;
			buttons.forEach( function ( b ) {
				var k = b.getAttribute( 'data-key' );
				b.disabled = true;
				if ( k === q.correct ) { b.className += ' krd-right'; } else if ( k === key ) { b.className += ' krd-wrong'; } else { b.className += ' krd-dim'; }
			} );
			var ok = key === q.correct;
			var ex = el( 'div', 'margin:10px 0 6px;padding:10px 14px;background:#fff;border-right:4px solid ' + ( ok ? '#34a853' : '#d93025' ) +
				';border-radius:8px;line-height:1.6', '<b>' + ( ok ? '✅ נכון!' : '❌ לא בדיוק. התשובה הנכונה: ' + esc( q.correct ) + '.' ) + '</b> ' + q.explain );
			ex.className = 'krd-fade';
			box.appendChild( ex );
			var last = cur === qs.length - 1;
			var next = el( 'button', 'margin-top:6px', last ? 'לתוצאות ›' : 'השאלה הבאה ›' );
			next.className = 'krd-btn';
			next.onclick = function () {
				if ( last ) { results(); } else { cur++; question(); }
				top();
			};
			box.appendChild( next );
		}

		function results() {
			var score = answers.filter( function ( a, i ) { return a === qs[ i ].correct; } ).length;
			stage.innerHTML = '';
			var box = el( 'div' );
			box.className = 'krd-fade';
			box.appendChild( el( 'div', 'text-align:center;margin:6px 0 4px', '<span style="font-size:260%;font-weight:bold;color:' + BROWN + '">' +
				score + ' / ' + qs.length + '</span>' ) );
			box.appendChild( el( 'div', 'text-align:center;font-size:110%;margin-bottom:12px', verdict( score, qs.length ) ) );

			var list = el( 'div', '' );
			qs.forEach( function ( q, i ) {
				var ok = answers[ i ] === q.correct;
				var mine = q.opts.filter( function ( o ) { return o.key === answers[ i ]; } )[ 0 ];
				var right = q.opts.filter( function ( o ) { return o.key === q.correct; } )[ 0 ];
				var row = el( 'div', 'background:#fff;border:1px solid #e7dcc8;border-right:5px solid ' + ( ok ? '#34a853' : '#d93025' ) +
					';border-radius:8px;padding:8px 12px;margin:6px 0;line-height:1.5' );
				row.innerHTML = '<div style="font-size:82%;color:#8a6a5c">' + ( ok ? '✅' : '❌' ) + ' ' + esc( q.yearLabel ) + '</div>' +
					'<div style="font-weight:bold">' + q.html + '</div>' +
					( ok ? '<div>ענית: ' + ( right ? right.html : esc( q.correct ) ) + '</div>' :
						'<div style="color:#b3261e">ענית: ' + ( mine ? mine.html : '' ) + '</div>' +
						'<div style="color:#1e7e34">התשובה הנכונה: ' + ( right ? right.html : esc( q.correct ) ) + '</div>' );
				list.appendChild( row );
			} );
			box.appendChild( list );

			var again = el( 'button', 'margin:12px 0 4px', '↺ משחק חוזר' );
			again.className = 'krd-btn krd-btn-light';
			again.onclick = start;
			box.appendChild( again );
			stage.appendChild( box );

			// ציר הזמן הגרפי, הכרטיסים והסיפור
			if ( events.length ) {
				var era = eraTimeline( periods, events, min, max, root.clientWidth - 60 );
				era.className = 'krd-era krd-fade';
				$timeline.show().find( '.krd-timeline-title' ).after( era );
				decorateEvents( events );
			}
			$story.show();
		}

		intro();
	}

	mw.loader.using( 'mediawiki.util' ).then( function () {
		mw.util.addCSS( CSS );
		mw.hook( 'wikipage.content' ).add( function ( $c ) {
			$c.find( '.krd-quiz' ).each( function () { init( this ); } );
		} );
	} );
}() );
