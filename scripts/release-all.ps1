# Bump each package version and push release tags.
#
# Usage:
#   .\scripts\release-all.ps1                          # patch bump (0.1.0 -> 0.1.1)
#   .\scripts\release-all.ps1 -Bump minor              # minor bump (0.1.0 -> 0.2.0)
#   .\scripts\release-all.ps1 -Bump major              # major bump (0.1.0 -> 1.0.0)
#   .\scripts\release-all.ps1 -m "my commit message"
param(
    [ValidateSet("patch","minor","major")]
    [string]$Bump = "patch",
    [Alias("m")]
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"

$Packages = @(
    "rai-check-core"
    "rai-check-ml"
    "rai-check-dl"
    "rai-check-genai"
    "rai-check-kit"
)

function Invoke-Git([string[]]$GitArgs) {
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Bump-Version($version, $bump) {
    $parts = $version -split '\.'
    switch ($bump) {
        "major" { return "$([int]$parts[0] + 1).0.0" }
        "minor" { return "$($parts[0]).$([int]$parts[1] + 1).0" }
        "patch" { return "$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)" }
    }
}

$status = Invoke-Git -GitArgs @("status", "--porcelain")
if ($status) {
    throw "Working tree is not clean. Commit or stash changes before releasing."
}

$NewVersions = @{}

Write-Host "Bumping versions ($Bump):"
foreach ($pkg in $Packages) {
    $toml = "packages/$pkg/pyproject.toml"
    $content = Get-Content $toml -Raw
    if ($content -match 'version = "([^"]+)"') {
        $current = $Matches[1]
    } else {
        Write-Error "Could not find version in $toml"
    }
    $new = Bump-Version $current $Bump
    $NewVersions[$pkg] = $new
    $content = $content -replace "version = `"$current`"", "version = `"$new`""
    Set-Content $toml $content -NoNewline
    Write-Host "  $pkg`: $current -> $new"
}

$commitMsg = if ($Message) { $Message } else { "chore: bump versions ($Bump)" }

Write-Host ""
Write-Host "Committing..."
Invoke-Git -GitArgs @("add", "-A")
Invoke-Git -GitArgs @("commit", "-m", $commitMsg)

Write-Host "Tagging..."
foreach ($pkg in $Packages) {
    $tag = "$pkg-v$($NewVersions[$pkg])"
    Invoke-Git -GitArgs @("tag", $tag)
    Write-Host "  Tagged $tag"
}

Invoke-Git -GitArgs @("push", "origin", "main")

# GitHub does not emit tag push events when more than three tags are pushed at once.
# Push each tag separately so each package publish workflow is triggered.
Write-Host "Pushing tags..."
foreach ($pkg in $Packages) {
    $tag = "$pkg-v$($NewVersions[$pkg])"
    Invoke-Git -GitArgs @("push", "origin", $tag)
}

Write-Host ""
Write-Host "Done - $($Packages.Count) publish workflows triggered."
