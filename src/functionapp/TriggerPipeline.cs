using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace functionapp
{
    public static class TriggerPipeline
    {
        [FunctionName("TriggerPipeline")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            var org = Environment.GetEnvironmentVariable("DevOpsOrganization");
            var project = Environment.GetEnvironmentVariable("DevOpsProject");
            var tokenName = Environment.GetEnvironmentVariable("DevOpsTokenName");
            var tokenValue = Environment.GetEnvironmentVariable("DevOpsTokenValue");

            var definitionId = 0;
            Int32.TryParse(req.Query["definitionId"], out definitionId);

            using(HttpClient client = new HttpClient())
            {
                client.DefaultRequestHeaders.Accept.Clear();
                client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                var token = Convert.ToBase64String(System.Text.ASCIIEncoding.ASCII.GetBytes(string.Format("{0}:{1}", tokenName, tokenValue)));
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", token);

                var bodyJson = $"{{'definitionId':{definitionId},'isDraft':false,'reason':'none','manualEnvironments':null}}";
                var bodyContent = new StringContent(bodyJson, Encoding.UTF8, "application/json");
                var response = await client.PostAsync($"https://vsrm.dev.azure.com/{org}/{project}/_apis/release/releases?api-version=7.1-preview.8", bodyContent);
                response.EnsureSuccessStatusCode();
            }

            return new OkObjectResult("OK");
        }
    }
}
