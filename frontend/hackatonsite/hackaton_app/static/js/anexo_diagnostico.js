angular.module('hackatonApp').controller('anexo_diagnostico', function($scope, $http, $interval,$filter, fileUpload) {
	
	$scope.checkExtension = function(file){
		var check = false;
		var extensionAux = file.split(".");
		var length = extensionAux.length;
		var extensionFile = extensionAux [length-1]
		
		if (extensionFile == 'pdf' || extensionFile == 'docx'){
			check = true;
		}
		return [check,extensionFile]
	};
	
	$scope.uploadFile = function(){
		var file = $scope.myFile;
		
		if ($scope.checkExtension(file.name)[1] === 'pdf'){
			var uploadUrl = "/CodeMyHealth/obtain_text_from_pdf_file/";
		}else if ($scope.checkExtension(file.name)[1] === 'docx') {
			var uploadUrl = "/CodeMyHealth/obtain_text_from_docx_file/";
		}
		//var uploadUrl = "/CodeMyHealth/obtain_text_from_doc_file/";
		
		if ($scope.myFile === undefined){
			swal("Campo vacío", "Archivo no seleccionado", "error")
		}else if (!$scope.checkExtension(file.name)[0]){
			swal("Extensión incorrecta", "Por favor introduzca un archivo con la extensión correcta ('.pdf','.docx')", "error")
		}else{
			$("#progressMaintenance").css("display","block");
			$(".content").css("opacity","0.3");
			//call to service
			fileUpload.uploadFileToUrl(file, uploadUrl).then(function(data){
				if (data.data.error !== ""){
					$("#progressMaintenance").css("display","none");
					$(".content").css("opacity","1");
					swal("Failed request", data.data.error, "error");
				}else{
					$scope.dataPDF = data.data.response
					$("#progressMaintenance").css("display","none");
					$(".content").css("opacity","1");
				}
			})
		}
		$scope.myFile = "";
		$(".fileUpload").filestyle("clear");
	};
	
});