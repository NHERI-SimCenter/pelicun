document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('div.nboutput.docutils.container').forEach(function(div) {
        // Our objective is to hide all `div` elements of which all
        // children elements only contain whitespace.
        // This remedies the nbsphinx issue where an extra newline was
        // added to each line in the code block output.
        let isEmpty = Array.from(div.children).every(child => !child.textContent.trim());
        
        if (isEmpty) {
            div.style.display = 'none';
        }
    });
});
