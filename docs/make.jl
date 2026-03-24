using AlternatingCrossInterpolation
using Documenter

DocMeta.setdocmeta!(AlternatingCrossInterpolation, :DocTestSetup, :(using AlternatingCrossInterpolation); recursive=true)

makedocs(;
    modules=[AlternatingCrossInterpolation],
    authors="Marc Ritter <mritter@flatironinstitute.org> and contributors",
    sitename="AlternatingCrossInterpolation.jl",
    format=Documenter.HTML(;
        canonical="https://rittermarc.github.io/AlternatingCrossInterpolation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rittermarc/AlternatingCrossInterpolation.jl",
    devbranch="main",
)
