function getImage() {
    $('#loader').removeClass('disabled').addClass('active');
    document.getElementById("formsubmit").submit();    
};

function getVideo() {
    const fi = document.getElementById('hiddeninputvideo'); 
    if (fi.files.length > 0) { 
        for (const i = 0; i <= fi.files.length - 1; i++) { 
  
            const fsize = fi.files.item(i).size; 
            const file = Math.round((fsize / 1024)); 
            const max_file_size = 30720;
            if (file >= max_file_size) { 
                alert("File too Big, please select a file less than 30mb (10 sec at 1080p or 5 sec at 4k)"); 
            } else { 
                $('#loader').removeClass('disabled').addClass('active');
                document.getElementById("formsubmit").submit();
            } 
        } 
    }      
};

function uploadImage() {
    $('#hiddeninputfile').click(); 
}

function uploadVideo() {
    $('#hiddeninputvideo').click(); 
}