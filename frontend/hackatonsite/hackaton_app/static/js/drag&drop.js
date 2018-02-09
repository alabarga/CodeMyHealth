function allowDrop (ev){
	ev.preventDefault();
}

function drag(ev) {
	ev.dataTransfer.setData("text", ev.target.id);
}

//adaptado para mozzila y chrome
function drop(ev) {
	
	ev.preventDefault();
	var data = ev.dataTransfer.getData("text");
	
	if (document.getElementById(data)){
		var idPadreElementoMovible = document.getElementById(data).parentNode.id
		if(idPadreElementoMovible == "elementSeleccionado" || idPadreElementoMovible == "elementNoSeleccionado"){
			if(ev.target){
				var idDestino = ev.target.id;
				if(idDestino != "elementSeleccionado" && idDestino != "elementNoSeleccionado"){
					if(idPadreElementoMovible != ev.target.parentNode.id){
						ev.target.parentNode.appendChild(document.getElementById(data));
					}
				}else{
					if(idPadreElementoMovible != idDestino){
						ev.target.appendChild(document.getElementById(data));
					}
				}
			}
		}
	}
}