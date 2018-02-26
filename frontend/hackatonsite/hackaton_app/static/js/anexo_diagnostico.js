angular.module('hackatonApp').controller('anexo_diagnostico', function($scope, $http, $interval,$filter, fileUpload, $rootScope) {
	
	$scope.enEjecucion = 0;
	
	$scope.checkExtension = function(file){
		var check = false;
		var extensionAux = file.split(".");
		var length = extensionAux.length;
		var extensionFile = extensionAux [length-1]
		
		if (extensionFile == 'pdf' || extensionFile == 'docx' || extensionFile == 'txt' || extensionFile == 'csv'){
			check = true;
		}
		return [check,extensionFile]
	};
	
	$scope.uploadFile = function(){
		var file = $scope.myFile;
		
		var myText = angular.element(document.querySelector('.textCie'));
		myText.text('El IR devuelve las siguientes sugerencias:');
		
		if ($scope.myFile == undefined || $scope.myFile == ""){
			swal("Archivo requerido", "Debe introducir un diagnóstico para su clasificación.", "error")
		}else if (!$scope.checkExtension(file.name)[0]){
			swal("Extensión incorrecta", "Por favor introduce un archivo con la extensión correcta ('.pdf', '.docx', '.txt', '.csv')", "error")
		}else{
			if ($scope.checkExtension(file.name)[1] === 'pdf'){
				var uploadUrl = "/CodeMyHealth/obtain_text_from_pdf_file/";
			}else if ($scope.checkExtension(file.name)[1] === 'docx') {
				var uploadUrl = "/CodeMyHealth/obtain_text_from_docx_file/";
			}else if ($scope.checkExtension(file.name)[1] === 'txt' || $scope.checkExtension(file.name)[1] === 'csv') {
				var uploadUrl = "/CodeMyHealth/obtain_text_from_text_file/";
			}
			$scope.mostrar(1)
			$scope.enEjecucion += 1
			//call to service
			fileUpload.uploadFileToUrl(file, uploadUrl, $rootScope.country).success(function(data){
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
			})
		}
		$scope.myFile = "";
		$(".fileUpload").filestyle("clear");
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