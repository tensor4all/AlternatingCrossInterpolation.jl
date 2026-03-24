using AlternatingCrossInterpolation
using Documenter

DocMeta.setdocmeta!(AlternatingCrossInterpolation, :DocTestSetup, :(using AlternatingCrossInterpolation); recursive=true)

makedocs(;
    modules=[AlternatingCrossInterpolation],
    authors="Marc Ritter <mritter@flatironinstitute.org> and contributors",
    sitename="AlternatingCrossInterpolation.jl",
    format=Documenter.HTML(;
        canonical="https://tensor4all.github.io/AlternatingCrossInterpolation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tensor4all/AlternatingCrossInterpolation.jl",
    devbranch="main",
)
