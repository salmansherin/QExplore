
var safe_require = function(locale, provider) {
	return localeRequires[locale][provider] || {};
};

