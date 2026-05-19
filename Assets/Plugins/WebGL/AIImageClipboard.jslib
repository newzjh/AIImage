mergeInto(LibraryManager.library, {
  AIImageClipboardCopyPNGFromUnity: function(ptr, len) {
    try {
      var bytes = HEAPU8.slice(ptr, ptr + len);
      var blob = new Blob([bytes], { type: 'image/png' });
      if (navigator.clipboard && navigator.clipboard.write && typeof ClipboardItem !== 'undefined') {
        var item = new ClipboardItem({ 'image/png': blob });
        navigator.clipboard.write([item]);
      }
    } catch (e) {}
  }
});

