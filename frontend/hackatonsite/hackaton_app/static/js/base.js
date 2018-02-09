var app = angular.module('hackatonApp', ["ui.bootstrap",'ngSanitize']);


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
	this.uploadFileToUrl = function(file, uploadUrl){//meter id
		var message = null;
		var fd = new FormData();
		fd.append('file', file);
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
}]);

/************************* FIN DOWNLOAD/UPLOAD FUNCTIONS ***********************************/