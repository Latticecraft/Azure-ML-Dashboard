using System;
using System.IO;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Storage;
using Microsoft.Azure.WebJobs.Host;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Microsoft.Extensions.Logging;

namespace functions
{
    public class UpdateMetadata
    {
        [FunctionName("UpdateMetadata")]
        public void Run([BlobTrigger("output/{name}", Connection = "AzureWebJobsStorage")]BlobClient myBlob, string name, ILogger log)
        {
            log.LogInformation($"C# Blob trigger function Processed blob\n Name:{name}");

            if (name.EndsWith(".html") == true)
            {
                // Get the existing properties
                var properties = myBlob.GetProperties().Value;
                if (properties.ContentType != "text/html")
                {
                    var headers = new BlobHttpHeaders
                    {
                        // Set the MIME ContentType every time the properties 
                        // are updated or the field will be cleared
                        ContentType = "text/html",
                        CacheControl = "max-age=3600",

                        // Populate remaining headers with 
                        // the pre-existing properties
                        ContentDisposition = properties.ContentDisposition,
                        ContentEncoding = properties.ContentEncoding,
                        ContentHash = properties.ContentHash,
                        ContentLanguage = properties.ContentLanguage
                    };

                    // Set the blob's properties.
                    myBlob.SetHttpHeaders(headers);
                }
            }
        }
    }
}
