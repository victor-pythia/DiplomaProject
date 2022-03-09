function newGame() {
      $.get('/newgame', function(r) {
        document.querySelector('p').innerText = '';
        board.position(r);
      });
}