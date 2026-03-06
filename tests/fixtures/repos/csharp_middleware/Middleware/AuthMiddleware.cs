using Microsoft.AspNetCore.Http;
using System.IdentityModel.Tokens.Jwt;
using System.Threading.Tasks;

namespace WebApp.Middleware
{
    public class AuthMiddleware
    {
        private readonly RequestDelegate _next;

        public AuthMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            var authHeader = context.Request.Headers["Authorization"].ToString();
            if (string.IsNullOrEmpty(authHeader) || !authHeader.StartsWith("Bearer "))
            {
                context.Response.StatusCode = 401;
                await context.Response.WriteAsync("Unauthorized");
                return;
            }

            var token = authHeader.Substring(7);
            var handler = new JwtSecurityTokenHandler();
            var jwt = handler.ReadJwtToken(token);
            context.Items["UserId"] = jwt.Subject;
            context.Items["Roles"] = jwt.Claims
                .Where(c => c.Type == "role")
                .Select(c => c.Value)
                .ToList();

            await _next(context);
        }
    }
}
