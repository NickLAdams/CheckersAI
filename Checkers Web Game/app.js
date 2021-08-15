const square0 = document.getElementById("0")

function updateBoard() {
    if (square0.innerHTML == '<span class="Wdot"></span>') {
       square0.innerHTML = '<span class="Bdot"></span>'; 
    }
    else {
        square0.innerHTML = '<span class="Wdot"></span>';
    }
}
