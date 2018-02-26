var app = angular.module('hackatonApp', ["ui.bootstrap",'ngSanitize']);

angular.module('hackatonApp').controller('mainController', function($scope, $http, $interval,$filter, fileUpload, $rootScope) {
	
	$scope.changeCountry = function(country) {
		if (country == 'cat'){
			$(".imagenCataluna").css("opacity","1");
			$(".imagenEspana").css("opacity","0.5");
			$scope.changeText(country)
			
		}else if (country == 'esp'){
			$(".imagenCataluna").css("opacity","0.5");
			$(".imagenEspana").css("opacity","1");
			$scope.changeText(country)
		}
		$rootScope.country = country
	};
	
	$scope.changeText = function(lang) {
		if ($('.bodyTable tr')){
			$('.bodyTable tr').each(function() {
				$(this).find("td").eq(1).html( $scope.obtenerTextoActualizado( $(this).find("td").eq(0).html(), lang ));
			});
		}
		if ($('.bodyTableIR tr')){
			$('.bodyTableIR tr').each(function() {
				$(this).find("td").eq(1).html( $scope.obtenerTextoActualizado( $(this).find("td").eq(0).html(), lang ));
			});
		}
	};
	
	$scope.obtenerTextoActualizado = function(codigo, lang) {
		cie_languages={}
		cie_languages['I00-I02'] = {'esp': 'Fiebre reumática aguda', 'cat': 'Febre reumàtica aguda'}
		cie_languages['I05-I09'] = {'esp': 'Enfermedades reumáticas crónicas cardiacas', 'cat': 'Cardiopaties reumàtiques cròniques'}
		cie_languages['I10-I16'] = {'esp': 'Enfermedades hipertensivas', 'cat': 'Malalties hipertensives'}
		cie_languages['I20-I25'] = {'esp': 'Enfermedades isquémicas cardíacas', 'cat': 'Cardiopaties isquèmiques'}
		cie_languages['I26-I28'] = {'esp': 'Enfermedad cardiaca pulmonar y enfermedades de la circulación pulmonar', 'cat': 'Malaltia cardiopulmona i malalties de la ciculació pulmonar'}
		cie_languages['I30-I52'] = {'esp': 'Otras formas de enfermedad cardiaca', 'cat': 'Altres formes de cardiopatia'}
		cie_languages['I60-I69'] = {'esp': 'Enfermedades cerebrovasculares','cat': 'Malalties cerebrovasculars'}
		cie_languages['I70-I79'] = {'esp': 'Enfermedades de arterias, arteriolas y capilares', 'cat': "Malalties d'artèries, arterioles i capil·lars"}
		cie_languages['I80-I89'] = {'esp': 'Enfermedades de vena, vasos linfáticos y nodos linfáticos, no clasificados en otra parte', 'cat': 'Malalties de venes, vasos limfàtics i ganglis limfàtics no classificades a cap altre lloc'}
		cie_languages['I95-I99'] = {'esp': 'Otros trastornos del sistema circulatorio y trastornos sin especificar', 'cat': "Altres trastorns de l'aparell circulatori i trastorns de l'aparell circularoti no especificats"}
		
		return cie_languages[codigo][lang]
	}
	
	
	$scope.getClass = function getClass(item) {
		var item_score = $scope.strip(item['score'].replace('Menor que ',''));
		if (parseInt(item_score) >= 80){
			return "specialOne"
		}else if (parseInt(item_score) >= 40 && parseInt(item['score']) < 80){
			return "specialTwo"
		}else if (parseInt(item_score) >= 10 && parseInt(item['score']) < 40){
			return "specialThree"
		}else {
			return "noSpecial"
		}
	};
	
	$scope.strip = function(texto) {
		return texto.replace(/^\s+|\s+$/g, '')
	};
	
	$scope.mostrar = function(number){
		if (number==1){
			$("#progressMaintenance").css("display","block");
			$(".tablas").css("opacity","0.3");
			$(".secondModel").css("display","none");
		}
		if (number==2){
			$("#progressMaintenance").css("display","none");
			$(".tablas").css("opacity","1");
			$(".secondModel").css("display","none");
		}
		if (number==3){
			$("#progressMaintenance").css("display","none");
			$(".tablas").css("opacity","1");
		}
		if (number==4){
			$(".tablaIR").css("display","none");
			$(".textCie").css("display","none");
			$(".secondModel").css("display","block");
		}
		if (number==5){
			$(".tablaIR").css("display","block");
			$(".textCie").css("display","block");
			$(".secondModel").css("display","none");
		}
		if (number==6){
			$(".secondModel").css("display","none");
			$(".tablaIR").css("display","block");
			$(".textCie").css("display","block");
		}
	};
	
	
	
});

/************************* DOWNLOAD/UPLOAD FUNCTIONS ***********************************/
app.directive('fileModel', ['$parse', function ($parse) {
	return {
		restrict: 'A',
		link: function(scope, element, attrs) {
			var model = $parse(attrs.fileModel);
			var modelSetter = model.assign;
			
			element.bind('change', function(){
				scope.$apply(function(){
					modelSetter(scope, element[0].files[0]);
				});
			});
		}
	};
}]);


//services for download/upload files
app.service('fileUpload', ['$http', function ($http) {
	//Service to upload file
	this.uploadFileToUrl = function(file, uploadUrl, lang){
		var message = null;
		var fd = new FormData();
		fd.append('file', file);
		fd.append('lang', JSON.stringify(lang));
		return $http({
				method: 'POST',
				url: uploadUrl,
				headers: {'Content-Type': undefined},
				data: fd,
				transformRequest: angular.identity
			})
	}

	//Service to download file
	this.downloadFileToUrl = function(nameDownload, download){
		$.fileDownload(download + '?nameDownload='+ nameDownload, {
			 successCallback : function(url) {
			 },
			 failCallback : function(html, url) {
			 }
	});
	}
	
	
	this.languaje = function(lang){
		$.fileDownload(download + '?nameDownload='+ nameDownload, {
			 successCallback : function(url) {
			 },
			 failCallback : function(html, url) {
			 }
	});
	}
}]);

/************************* FIN DOWNLOAD/UPLOAD FUNCTIONS ***********************************/

app.run(function($rootScope) {
	$rootScope.country = 'esp';
});