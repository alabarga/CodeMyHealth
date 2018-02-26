angular.module('hackatonApp').controller('redactar_diagnostico', function($scope, $http, $interval,$filter, fileUpload, $rootScope) {

	//variables globales
	$scope.enEjecucion = 0;
	
	$scope.comprobar = function (data){
		$scope.messageValidation = "";
		var errores = [];
		if (data['CIE'] === null || data['CIE'] === "")
			errores.push("Debe introducir un diagnóstico para su clasificación.");
		if( errores.length > 0 ){
			var separador = "";
			$.each(errores, function(key, value){
				$scope.messageValidation += separador + value;
				separador = "\n";
			})
		}
	}
	
	$scope.getDiagnostico = function() {
		
		$scope.diagnostico = '';//inicializacion de variables
		$scope.diagnostico = angular.element( document.querySelector( '#cie' ) ); //Coge elemento por id mediante DOM
		
		var myText = angular.element(document.querySelector('.textCie'));
		myText.text('El IR devuelve las siguientes sugerencias:');

		var data = {
			"CIE" : $scope.diagnostico.context.value,
			'lang': $rootScope.country
        }
		$scope.comprobar(data);
		if($scope.messageValidation == ""){
			$scope.mostrar(1)
			$scope.enEjecucion += 1
			var response = $http.post("/CodeMyHealth/getDiagnostico/", data);
			response.success(function(data, status, headers, config) {
				if (data.error !== ""){
					$scope.mostrar(2)
					$scope.enEjecucion -= 1
					swal("Servicio caído", data.error, "error");
				}else{
					$scope.enEjecucion -= 1
					if ($scope.enEjecucion < 1){
						$scope.data = data.response

						$scope.textDiagnostico = $scope.data['Modelo1']
						$scope.getIR($scope.data) //vemos a ver que hacer con el caso 2
						
						$scope.mostrar(3)
						$(".tablas").css("display","block");
					}
				}
				
			}).error(function(data, status) {
				console.error('Repos error', status, data);
			});
		}else{
			swal("Texto requerido", $scope.messageValidation, "error");
		}
	};
	
	$scope.getIR = function(data) {
		if ('Error' in data){
			var myText = angular.element(document.querySelector('.textCie'));
			myText.text('El IR no devuelve ninguna sugerencia');
			$(".tablaIR").css("display","none");
		}else{
			if (parseInt(data['Modelo1'][0]['score']) >= 80){
				$scope.mostrar(4)
			}else{
				$scope.mostrar(5)
			}
			$scope.textIR = data['Modelo2']
		}
	};
	
});