// todo: precedence: 
//             enable parsing of 2a (latex output 2a) also -a  
//             fix output like 10+-3 

//              build a polimorphic function for complete RTTI? 

//              build a more generic regression testing framework? (a one that can execute any function that return a boolean (fail/pass value)

// the colne of PolyExpression:I needed to explictily list all the field copies.. is there a better way to do it in js? 

// next step: add unary -, make all "a-b" --> a + - b but Tex it as a-b ...   

// step: transform this to a ploynom form: (x-56)*(x-34)=0


// globals 
latexUrl = "http://www.forkosh.dreamhost.com/mathtex.cgi?";


operators = [['='],['+','-'],['*','/'],['^']]; // groups and order according to 
                                           // precedence

precedence = [];  // will hold a map char->int

FAIL_VAL = -17; 

wIp = 0; // write pointer 

function init() {

    for(var i = 0; i < operators.length; ++i) {
	var g = operators[i]; 
	for(var j=0; j<g.length; ++j) 
	    precedence[g[j]] = i; 
    }
}

function handleInputString(e) {
    init(); // probably should be called from elsewhere in the html..  
    
    // only if enter was pressed 
    if ( (e.which && e.which != 13) || (window.event && e.keyCode != 13))  //the second condition for IE bug  
        return;

    var eqStr = document.getElementById("equationInput").value; 

    var e = parseStr(eqStr);
    var e1 = e.derive(); 
//    flatten(e); 
    write("> "+e1.toStr());
//    write(e.toTex());
    texWrite(e1.toTex()); 
//    regressionTests();
}

// --- parser code --- 

function parseStr(str) {
    return parseEquation(str,0,str.length);
}

function parseEquation(str,bi,ei) { 
    // bi - begin index, ei - one-past-end index 
    
    if ( str[ei-1] == ')' && (jumpOverParenthesis(str,bi,ei-1) == bi-1))
	return parseEquation(str,bi+1,ei-1); // strip paren
    
    var nextOp = findTopOperator(str,bi,ei);

    var j = nextOp[0]; 
    var op = nextOp[1];
    
    if (op == '') { // no top operator 
	if (ei-bi == 1) { // single character case 
	    c = str[bi]; 
	    if (isParameter(c)) 
		return new Parameter(c);
	    if (isVariable(c))
		return new Variable(c); 
	}

	var v = parseFloat(str.slice(bi,ei));  

	if(isNaN(v)) {
	    alert(str.slice(bi,ei) + " is not a valid expression"); 
	    return new Expression(); 
	}
	
	return new Number(v); 
    }

    var je = j; 
    if(op == 'X') { // 'X' denotes a mul operation with no sign  
	je = j+1; 
	op = '*'; 
    }
    var r1 = parseEquation(str,bi,je);
    var r2 = parseEquation(str,j+1,ei);
    
    var exp;
    
    switch(op) {
    case '+':
	exp = new PlusExpression();
	break;
    case '-':
	exp = new MinusExpression();
	break;
    case '*':
	exp = new MulExpression();
	break;
    case '/':
	exp = new DivExpression();
	break;
    case '^':
	exp = new PowExpression();
	break;
    case '=':
	exp = new EquationExpression();
	break;
	
    default:
	alert("unkown operator "+op);
	return new Expression(); 
    }

    exp.addElement(r1);
    exp.addElement(r2);
    return exp; 
}

// params: "abcdefgh", no use: "ijklmnopq", variables:  "rstuvwxyz" 

function isParameter(c) {
    return (c >= 'a' && c <= 'h'); 
}

function isVariable(c) {
    return (c >= 'r' && c <= 'z') ||  (c >= 'R' && c <= 'Z');
}

function jumpOverParenthesis(str,bi,i) {
    var c = 1; 
    --i; 
    while(c > 0) {
	if (i < bi) {
	    alert("Unbalanced parenthesis"); 
	    return FAIL_VAL; 
	} 
	if(str[i] == ')')
	    ++c;
	if(str[i] == '(')
	    --c; 
	--i;
    }
    return i; 
}

function isNumeric(c) {
    return c>= '0' && c <= '9' || c == '.'; 
}

function isAlpha(c) {
    return c>='A' && c <= 'Z' || c>='a' && c <= 'z'; 
}

function isNumericOrAlpha(c) {
    return isNumeric(c) || isAlpha(c); 
}

function isMulWithoutSign(s1,s2) {
    return isAlpha(s1) && isNumeric(s2) ||
	isNumeric(s1) && isAlpha(s2) || 
	isNumericOrAlpha(s1) && s2 == '(' ||
	s1 == ')' && isNumericOrAlpha(s2); 
}

function findTopOperator(str,bi,ei) {
    for(var p=0; p < operators.length; ++p) { // loop over ops groups 
	var i = ei-1;
	var g = operators[p];
	while (i >= bi) { 

	    for(var j=0; j<g.length; ++j) { 
		if(str[i] == g[j] && i>bi)
		    return [i, str[i] ]; // found top op
		if(g[j] == '*' && 
		   i < ei-1 && 
		   isMulWithoutSign(str[i],str[i+1])) 
		    return [i,'X']; // found mul operation without a sign  
	    }  

	    if(str[i] == ')') { 
		i = jumpOverParenthesis(str,bi,i);
		if(i == FAIL_VAL) 
		    return [-1,'']; // Unbalanced parenthesis 
	    } else
		--i;
	} 
    }
    return [-1,'']; // no top operator found  
}

// --- end of parser code --- 
function pushArray(arr1,arr2) {
    for(var i=0; i<arr2.length; ++i)
	arr1.push(arr2[i]); 
}


function distribute(exp) { 
    // usage exp = distribute(exp) -- returns a cloned answer 
    var op = exp.op(); 

    if(op == null)
	return exp.clone(); 

    var els = exp.elements; 

    if(op == '*') { 
	for(var i=0; i<els.length; ++i) { // '*' loop 
	    var son = els[i];
	    if (son.op() == '+') {
		exp = new PlusExpression(); 
		var sels = son.elements;
		for(var j=0; j<sels.length; ++j) { // '+' loop 
		    var smul = new MulExpression(); 
		    
		    for(var k=0; k<i; ++k) 
			smul.elements[k] = els[k].clone();
		    
		    smul.elements[i] = sels[j].clone();
		    
		    for(var k=i+1; k<els.length; ++k) 
			smul.elements[k] = els[k].clone();
			
		    exp.addElement(smul);
		} // '+' loop 
		return distribute(exp);
	    }	// '+' case 
	} // '*' loop  
    } // '*' case
    // post order recursive activation 
    var exp = exp.nodeClone();
    var sonWasDistributed = false; 
    for(var i=0; i<els.length; ++i) {
	var oo = els[i].op();
	var ns = distribute(els[i]);
	exp.addElement(ns); 
	if(oo != '+' && ns.op() == '+')
	    sonWasDistributed = true; 
    }
    if (op == '*' && sonWasDistributed)
	return distribute(exp);
    return exp; 
}


function flatten(exp) {

    var op = exp.op(); 
    if(op == null)
	return;  

    var els = exp.elements; 
    var nels = [];
    for (var i=0; i<els.length; ++i) {
	var e = els[i];
	flatten(e);
	if((op == '+' || op == '*') && op == e.op()) 
	    pushArray(nels,e.elements)
	else
	    nels.push(e); 
    }
    exp.elements = nels; 
}

function write(s) {
    document.getElementById("out"+wIp).innerHTML = s+"<br />";
    wIp = (wIp+1)%4; 
}

function texWrite(s) {
    var theImg = document.getElementById("img1");
    theImg.src = latexUrl + s; 
}


function toTexWithParen(expU,expD,isFirst) {
    var opU = expU.op(); 
    var opD = expD.op();
    var expTex = expD.toTex(); 
 
    if (opD == 'null')
	return expTex; 

    if (precedence[opU] > precedence[opD] || 
	(!isFirst && opU == '-' && 
	 precedence[opU] == precedence[opD]) // case x-(x+x) 
       ) 
	return '('+expD.toTex()+')';
    return expTex;
}

// "Classes" 

function Expression() {
    this.toTex = null; 
    this.clone = null; 
    this.op = function() {
	return null; 
    }
}

function PolyExpression(opChar) {
    
    Expression.call(this); //inheritance 

    this.opChar = opChar
    this.opStr = opChar; 
    
    this.op = function() {
	return this.opChar;
    }

    this.clone = function() {
	var exp = new PolyExpression(this.opChar);
	exp.opStr = this.opStr;
	exp.toStr = this.toStr; 
	exp.toTex = this.toTex; // should I call nodeClone instead? 
	for(var i=0; i<this.elements.length; ++i)
	    exp.addElement(this.elements[i].clone()); 
	return exp; 
    }

    this.nodeClone = function() {
	var exp = new PolyExpression(this.opChar);
	exp.opStr = this.opStr;
	exp.toStr = this.toStr; 
	exp.toTex = this.toTex; 
	return exp; 
    }

    this.elements = new Array(); 

    this.toStr = function() {

	var str = this.opChar+"(";
	var last = this.elements.length-1
	for(var i = 0; i < last; ++i)
	    str += this.elements[i].toStr()+',';
	str += this.elements[last].toStr()+')'; 
	return str; 
    }

    this.toTex = function() {
	var str =  toTexWithParen(this,this.elements[0],true);
	for(var i = 1; i < this.elements.length; ++i)
	    str += this.opStr + toTexWithParen(this,this.elements[i],false);
	return str; 
    }

    this.addElement = function (exp) {
	this.elements.push(exp); 
    }
}

function PlusExpression() {
    PolyExpression.call(this,"+");  //inheritance
    this.derive = function() {
	var exp = new PlusExpression(); 
	for(var i=0; i<this.elements.length; ++i)
	    exp.addElement(this.elements[i].derive()); 
	return exp; 
    }
} 

function MinusExpression() {
    PolyExpression.call(this,"-");  //inheritance
} 

function MulExpression() {
    PolyExpression.call(this,"*");  //inheritance
    this.opStr = "\\cdot "; 

    this.derive = function() {
	// assume binary...
	var exp = new PlusExpression();
	var ls = new MulExpression();
	var rs = new MulExpression();
	exp.addElement(ls);
	exp.addElement(rs); 

	var l = this.elements[0];
	var r = this.elements[1];

	ls.addElement(l.derive());
	ls.addElement(r.clone()); 

	rs.addElement(l.clone()); 
	rs.addElement(r.derive());

	return exp;  
    }
} 

function DivExpression() {
    PolyExpression.call(this,"/");  //inheritance
    this.toTex = function() {
	if(this.elements.length != 2) {
	    alert("currently, only binary divide operation");
	    return "";
	}
	
	var str = "\\frac{" +  this.elements[0].toTex() + "}"; 
	str += "{" + this.elements[1].toTex() + "}";

	return str; 
    }
} 

function EquationExpression() {
    PolyExpression.call(this,"=");  //inheritance
} 

function PowExpression() {
    PolyExpression.call(this,"^");  //inheritance
    this.toTex = function() {
	if(this.elements.length != 2) {
	    alert("currently, only binary power operation");
	    return "";
	}
	
	var str = "{" + toTexWithParen(this,this.elements[0],true) + "}"; 
	str += "^{" +  this.elements[1].toTex() + "}";

	return str; 
    }
} 

function SymbolExp(c) {
    Expression.call(this); //inheritance
    this.symb = c; 
    this.toTex = function() { return this.symb; }
    this.toStr = this.toTex;  
    this.clone = function() { return new SymbolExp(this.symb); }
}

function Parameter(c) {
    SymbolExp.call(this,c); //inheritance
    this.derive = function() { return new Number(0); }
}

function Variable(c) {
    SymbolExp.call(this,c); //inheritance
    this.derive = function() { return new Number(1); }
}

function Number(n) {
    
    Expression.call(this); //inheritance

    this.value = n; 
    this.toTex = function() {
	return this.value;
    }
    this.toStr = this.toTex;  
    this.clone = function() { return new Number(this.value); }

    this.derive = function() { return new Number(0); }
} 


// == regression tests == 

test_instances = [    // ["input string", "expected structure"] 
    ["-88.88+(2+3)","+(-88.88,+(2,3))"], 
    ["1-(a+b)","-(1,+(a,b))"],
    ["a*(21+23)/23+(566-12)/(x^2)","+(/(*(a,+(21,23)),23),/(-(566,12),^(x,2)))"],
    ["a(1+1)b","*(*(a,+(1,1)),b)"],
    ["-28.33a+(55+3a)x","+(*(-28.33,a),*(+(55,*(3,a)),x))"],
    ["2x^2-45x+12=34x^34","=(+(-(*(2,^(x,2)),*(45,x)),12),*(34,^(x,34)))"]
];

function regressionTests() {
    var failed = 0;
    var num = test_instances.length; 
    for(var i=0; i<num; ++i) {
	var iStr = test_instances[i][0]; 
	var gStr = parseStr(iStr).toStr();
	var eStr = test_instances[i][1];
	if(gStr != eStr) {
	    write("test failed on string: " + 
		  iStr + "<br />expected: " + eStr + 
		  "<br />but got:&nbsp;&nbsp;&nbsp;&nbsp;" + gStr + "<br />");
	    ++failed; 
	}
    }
    if (failed == 0) 
	write("Ok, All the "+num+" tests have passed. All is fine");
    else 
	write("! "+failed + " tests have failed (out of "+num+")"+" !"); 
}

function addOneToAllNumbers(exp) {
    if(exp.op() == null) {
	if(typeof(exp.value) != 'undefined')
	    exp.value += 1; 
	return; 
    }
    for(var i=0; i<exp.elements.length; ++i)
	addOneToAllNumbers(exp.elements[i]);
}