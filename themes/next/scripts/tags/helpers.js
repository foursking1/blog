/* global hexo */

'use strict';

var pathFn = require('path');
var _ = require('lodash');

function startsWith(str, start) {
  return str.substring(0, start.length) === start;
}


hexo.extend.helper.register('doc_sidebar', function(className) {
  var type = this.page.canonical_path.split('/')[0];
  var sidebar = this.site.data.sidebar[type];
  var path = pathFn.basename(this.path);
  var result = '';
  var self = this;
  var prefix = 'sidebar.' + type + '.';


  _.each(sidebar, function(menu, title) {
    result += '<ol class="nav">'
    result += '<li class="nav-item"><span class="nav-title current"><strong>' + title + '</strong></span>';
    result += '<ol class="nav">'

    _.each(menu, function(link, text) {
      var itemClass = ""
      if (link === path) itemClass += ' current';
      result += '<li class="nav-item' + itemClass + '"><a href="' + link + '">' +  text + '</a></li>';
    });

    result += '</ol></li></ol>'


  });

  return result;
});


