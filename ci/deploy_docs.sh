#!/usr/bin/env bash

# Deploy organized HTML documentation to Github for hosting.
#
#    This script deploys organized HTML documentation produced by `prepare_docs.sh` to a repository in the MAST Team
#    GitHub repository for hosting using a simple git push. It is meant to be called during the Travis-CI
#    `deploy` stage and requires the GH_TOKEN environment variable be set in the Travis-CI job from
#    an account that can push to https://github.com/MASTmultiphysics/MASTmultiphysics.github.io. The GH_TOKEN variable
#    is specified in the Travis-CI web interface as a secret variable so it is not displayed publically.
#
#    Contents of this script (and `prepare_docs.sh`) were inspired in part
#    by https://gist.github.com/willprice/e07efd73fb7f13f917ea.

# Set git remote URL. Force push to GitHub website repository.
# Update git remote URL to include ${GH_TOKEN} key. Push files back to GitHub website repository.
cd ${TRAVIS_BUILD_DIR}/website || exit
git push --force origin master || exit

