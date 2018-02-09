angular.module('hackatonApp').controller('redactar_diagnostico', function($scope, $http, $interval,$filter, fileUpload) {
	
	$scope.comprobar = function (data){
		$scope.messageValidation = "";
		var errores = [];
		if (data['CIE'] === null || data['CIE'] === "")
			errores.push("El campo 'Diagnostico' es necesario.");
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

		var data = {
			"CIE" : $scope.diagnostico.context.value
        }
		$scope.comprobar(data);
		if($scope.messageValidation == ""){
			$("#progressMaintenance").css("display","block");
			$(".content").css("opacity","0.3");
			var response = $http.post("/CodeMyHealth/getDiagnostico/", data);
			response.success(function(data, status, headers, config) {
				if (data.error !== ""){
					$("#progressMaintenance").css("display","none");
					$(".content").css("opacity","1");
					swal("Failed request", data.error, "error");
				}else{
					$scope.textDiagnostico = data.response
					$("#progressMaintenance").css("display","none");
					$(".content").css("opacity","1");
				}
				
			}).error(function(data, status) {
				console.error('Repos error', status, data);
			});
		}else{
			swal("Campo vacío", $scope.messageValidation, "error");
		}
	};
	
});