use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{parse_macro_input, Fields, Item, ItemEnum, ItemStruct, Lifetime};

use crate::{common::*, derive_device_impl::device_impl_attributes::DeviceImplAttributes};

pub mod device_impl_attributes;

pub fn derive_device_impl(
    attrs: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attrs = parse_macro_input!(attrs as DeviceImplAttributes);
    let input = parse_macro_input!(input as Item);
    let impl_block = derive_device_impl_inner(attrs, &input);
    match impl_block {
        Ok(tokens) => quote! {
            #input
            #tokens
        },
        Err(e) => e.to_compile_error(),
    }
    .into()
}

pub fn derive_device_impl_inner(
    attrs: DeviceImplAttributes,
    input: &Item,
) -> syn::Result<TokenStream> {
    match input {
        Item::Struct(s) => derive_device_impl_struct(attrs, s),
        Item::Enum(e) => derive_device_impl_enum(attrs, e),
        _ => Err(syn::Error::new_spanned(
            input,
            "#[derive_device_impl] only works for structs or enums",
        )),
    }
}

fn derive_device_impl_struct(
    attrs: DeviceImplAttributes,
    input: &ItemStruct,
) -> syn::Result<TokenStream> {
    require_repr(&input.attrs)?;

    let device_name = &input.ident;
    let host_name = attrs.builder;

    let builder_lifetime = attrs.lifetime;

    let device_generics = &input.generics;
    let (host_generics, device_from_host_ty_generics) =
        host_generics_from_device(device_generics, &builder_lifetime, &attrs.extra_lifetime);

    // Process generics (preserve all parameters and where clauses)
    let (device_impl_generics, device_ty_generics, device_where_clause) =
        device_generics.split_for_impl();
    let (host_impl_generics, host_ty_generics, host_where_clause) = host_generics.split_for_impl();

    let actual_builder_lifetime = if let Some(lt) = &builder_lifetime {
        lt.clone()
    } else {
        Lifetime::new("'_", Span::mixed_site())
    };

    let fields = match &input.fields {
        Fields::Named(fields) => Ok(&fields.named),
        _ => Err(syn::Error::new_spanned(
            input.fields.clone(),
            "#[derive_device_impl] for structs only works with named fields",
        )),
    }?;

    let clone_fields = fields.iter().map(|field| {
        let field_name = &field.ident;
        quote! {
            #field_name: self.#field_name.clone()
        }
    });

    let impl_block = quote! {
        #[cfg(not(target_os = "cuda"))]
        unsafe impl #device_impl_generics cust::memory::DeviceCopy for #device_name #device_ty_generics #device_where_clause {}
        #[cfg(not(target_os = "cuda"))]
        impl #device_impl_generics Copy for #device_name #device_ty_generics #device_where_clause {}
        #[cfg(not(target_os = "cuda"))]
        impl #device_impl_generics Clone for #device_name #device_ty_generics #device_where_clause {
            fn clone(&self) -> Self {
                #device_name {
                    #(#clone_fields),*
                }
            }
        }
        impl #host_impl_generics From<&#device_name #device_from_host_ty_generics> for &#host_name #host_ty_generics #host_where_clause {
            fn from(device: &#device_name #device_from_host_ty_generics) -> Self {
                unsafe { &*(device as *const #device_name #device_from_host_ty_generics as *const #host_name #host_ty_generics) }
            }
        }
        impl #host_impl_generics From<&mut #device_name #device_from_host_ty_generics> for &mut #host_name #host_ty_generics #host_where_clause {
            fn from(device: &mut #device_name #device_from_host_ty_generics) -> Self {
                unsafe { &mut *(device as *mut #device_name #device_from_host_ty_generics as *mut #host_name #host_ty_generics) }
            }
        }

        #[cfg(target_os = "cuda")]
        impl #device_impl_generics gpu_builder::BuildResultType for #device_name #device_ty_generics #device_where_clause {}

        #[cfg(target_os = "cuda")]
        impl #host_impl_generics gpu_builder::Builder<#actual_builder_lifetime> for #host_name #host_ty_generics #host_where_clause {
            type Output = #device_name #device_from_host_ty_generics;
        }
    };

    Ok(TokenStream::from(impl_block))
}

fn derive_device_impl_enum(
    attrs: DeviceImplAttributes,
    input: &ItemEnum,
) -> syn::Result<TokenStream> {
    require_repr(&input.attrs)?;

    let device_name = &input.ident;
    let host_name = attrs.builder;

    let builder_lifetime = attrs.lifetime;

    let device_generics = &input.generics;
    let (host_generics, device_from_host_ty_generics) =
        host_generics_from_device(device_generics, &builder_lifetime, &attrs.extra_lifetime);

    // Process generics (preserve all parameters and where clauses)
    let (device_impl_generics, device_ty_generics, device_where_clause) =
        device_generics.split_for_impl();
    let (host_impl_generics, host_ty_generics, host_where_clause) = host_generics.split_for_impl();

    let actual_builder_lifetime = if let Some(lt) = &builder_lifetime {
        lt.clone()
    } else {
        Lifetime::new("'_", Span::mixed_site())
    };

    let variants_clone = make_variants(&input.variants, |field_names, make_enum| {
        let clone_fields = field_names
            .iter()
            .map(|field_name| {
                quote! {
                    #field_name.clone()
                }
            })
            .collect::<Vec<_>>();
        make_enum(device_name, clone_fields)
    });

    let impl_block = quote! {
        #[cfg(not(target_os = "cuda"))]
        unsafe impl #device_impl_generics cust::memory::DeviceCopy for #device_name #device_ty_generics #device_where_clause {}
        #[cfg(not(target_os = "cuda"))]
        impl #device_impl_generics Copy for #device_name #device_ty_generics #device_where_clause {}
        #[cfg(not(target_os = "cuda"))]
        impl #device_impl_generics Clone for #device_name #device_ty_generics #device_where_clause{
            fn clone(&self) -> Self {
                match self {
                    #(#variants_clone)*
                }
            }
        }
        impl #host_impl_generics From<&#device_name #device_from_host_ty_generics> for &#host_name #host_ty_generics #host_where_clause {
            fn from(device: &#device_name #device_from_host_ty_generics) -> Self {
                unsafe { &*(device as *const #device_name #device_from_host_ty_generics as *const #host_name #host_ty_generics) }
            }
        }
        impl #host_impl_generics From<&mut #device_name #device_from_host_ty_generics> for &mut #host_name #host_ty_generics #host_where_clause {
            fn from(device: &mut #device_name #device_from_host_ty_generics) -> Self {
                unsafe { &mut *(device as *mut #device_name #device_from_host_ty_generics as *mut #host_name #host_ty_generics) }
            }
        }

        #[cfg(target_os = "cuda")]
        impl #device_impl_generics gpu_builder::BuildResultType for #device_name #device_ty_generics #device_where_clause {}

        #[cfg(target_os = "cuda")]
        impl #host_impl_generics gpu_builder::Builder<#actual_builder_lifetime> for #host_name #host_ty_generics #host_where_clause {
            type Output = #device_name #device_from_host_ty_generics;
        }
    };

    Ok(TokenStream::from(impl_block))
}
